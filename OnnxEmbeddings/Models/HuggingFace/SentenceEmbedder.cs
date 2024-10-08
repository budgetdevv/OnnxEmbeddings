using System;
using System.Diagnostics;
using System.Linq;
using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxEmbeddings.Models.HuggingFace
{
    internal static class SentenceEmbedder
    {
        internal const string 
            INPUT_IDS = "input_ids",
            ATTENTION_MASK = "attention_mask",
            TOKEN_TYPE_IDS = "token_type_ids";
        
        public interface ISentenceEmbedderInputBase
        {
            static virtual string[] InputNames => [ INPUT_IDS, ATTENTION_MASK ];
        
            public long[] InputIDs { get; }
    
            public long[] AttentionMask { get; }
        
            public int[] Dimensions { get; }
        }
        
        // Exists to constrain the extension methods. ISentenceEmbedderInputExtended should not inherit from ISentenceEmbedderInput.
        public interface ISentenceEmbedderInput: ISentenceEmbedderInputBase { }
    
        public interface ISentenceEmbedderInputExtended: ISentenceEmbedderInputBase
        {
            static string[] ISentenceEmbedderInputBase.InputNames => [ INPUT_IDS, ATTENTION_MASK, TOKEN_TYPE_IDS ];
        
            public long[] TokenTypeIDs { get; }
        }
    
        public readonly struct Input: ISentenceEmbedderInput
        {
            public long[] InputIDs { get; init; }

            public long[] AttentionMask { get; init; }
        
            public int[] Dimensions { get; init; }

            public Input(long[] inputIDs, long[] attentionMask, int[] dimensions)
            {
                InputIDs = inputIDs;
                AttentionMask = attentionMask;
                Dimensions = dimensions;
            }

            public Input(string[] sentences, int maxSequenceLength, BertTokenizer wordPieceTokenizer)
            {
                var batchSize = sentences.Length;
                var bufferSize = batchSize * maxSequenceLength;

                var inputIDs = InputIDs = new long[bufferSize];
                var attentionMask = AttentionMask = new long[bufferSize];

                wordPieceTokenizer.Encode(sentences, inputIDs, attentionMask, maxSequenceLength);

                Dimensions = [ batchSize, maxSequenceLength ];
            }
        }
    
        public readonly struct InputExtended: ISentenceEmbedderInputExtended
        {
            public long[] InputIDs { get; init; }

            public long[] AttentionMask { get; init; }

            public long[] TokenTypeIDs { get; init; }
        
            public int[] Dimensions { get; init; }
            
            public InputExtended(long[] inputIDs, long[] attentionMask, long[] tokenTypeIDs, int[] dimensions)
            {
                InputIDs = inputIDs;
                AttentionMask = attentionMask;
                TokenTypeIDs = tokenTypeIDs;
                Dimensions = dimensions;
            }
            
            public InputExtended(string[] sentences, int maxSequenceLength, BertTokenizer wordPieceTokenizer)
            {
                var batchSize = sentences.Length;
                var bufferSize = batchSize * maxSequenceLength;

                var inputIDs = InputIDs = new long[bufferSize];
                var attentionMask = AttentionMask = new long[bufferSize];
                var tokenTypeIDs = TokenTypeIDs = new long[bufferSize];

                wordPieceTokenizer.Encode(sentences, inputIDs, attentionMask, tokenTypeIDs, maxSequenceLength);

                Dimensions = [ batchSize, maxSequenceLength ];
            }
        }
    
        public interface ISentenceEmbedderOutput
        {
            public static abstract string[] OutputNames { get; }

            public void PopulateOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output);
        }
        
        public struct LastHiddenStateOutput: ISentenceEmbedderOutput
        {
            public static string[] OutputNames => [ "last_hidden_state" ];

            public float[] LastHiddenState { get; set; }
        
            public void PopulateOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output)
            {
                LastHiddenState = ((DenseTensor<float>) output[0].Value).ToArray();
            }
        }
        
        public static OutputT GenerateEmbeddings<InputT, OutputT>(
            this SentenceEmbedder<InputT, OutputT> embedder,
            InputT input)
            where InputT: struct, ISentenceEmbedderInput
            where OutputT: struct, ISentenceEmbedderOutput
        {
            var dimensions = (ReadOnlySpan<int>) input.Dimensions.AsSpan();
            
            var inputIDsTensor = new DenseTensor<long>(input.InputIDs, dimensions);
            var attentionMaskTensor = new DenseTensor<long>(input.AttentionMask, dimensions);
            
            return embedder.GenerateEmbeddings(
            [
                NamedOnnxValue.CreateFromTensor(INPUT_IDS, inputIDsTensor),
                NamedOnnxValue.CreateFromTensor(ATTENTION_MASK, attentionMaskTensor),
            ]);
        }
        
        public static OutputT GenerateEmbeddings<InputT, OutputT>(
            this SentenceEmbedder<InputT, OutputT> embedder,
            InputT input,
            // ReSharper disable once MethodOverloadWithOptionalParameter
            bool _ = false)
            where InputT: struct, ISentenceEmbedderInputExtended
            where OutputT: struct, ISentenceEmbedderOutput
        {
            var dimensions = (ReadOnlySpan<int>) input.Dimensions.AsSpan();
            
            var inputIDsTensor = new DenseTensor<long>(input.InputIDs, dimensions);
            var attentionMaskTensor = new DenseTensor<long>(input.AttentionMask, dimensions);
            var tokenTypeIDsTensor = new DenseTensor<long>(input.TokenTypeIDs, dimensions);
            
            return embedder.GenerateEmbeddings(
            [
                NamedOnnxValue.CreateFromTensor(INPUT_IDS, inputIDsTensor),
                NamedOnnxValue.CreateFromTensor(ATTENTION_MASK, attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor(TOKEN_TYPE_IDS, tokenTypeIDsTensor),
            ]);
        }
    }
    
    internal sealed class SentenceEmbedder<InputT, OutputT>: IDisposable
        where InputT: struct, SentenceEmbedder.ISentenceEmbedderInputBase
        where OutputT: struct, SentenceEmbedder.ISentenceEmbedderOutput
    {
        private static readonly bool IS_EXTENDED_INPUT = typeof(InputT)
            .GetInterfaces()
            .Contains(typeof(SentenceEmbedder.ISentenceEmbedderInputExtended));
        
        private readonly SessionOptions SessionOptions;
        
        private readonly InferenceSession Session;
        
        private readonly string[] OutputNames;

        public SentenceEmbedder(string modelPath, SessionOptions? sessionOptions = null)
        {
            SessionOptions = sessionOptions ??= new();
            var session = Session = new(modelPath, sessionOptions);
            OutputNames = OutputT.OutputNames;
            
            Debug.Assert(
                condition: OutputNames.SequenceEqual(session.OutputMetadata.Keys.ToArray()),
                message: "The provided output names do not match the actual output names."
            );
        }

        public OutputT GenerateEmbeddings(NamedOnnxValue[] values)
        {
            using var runOptions = new RunOptions();
            
            // using var registration = cancellationToken.Register(() => runOptions.Terminate = true);

            using var runResult = Session.Run(values, OutputNames, runOptions);

            OutputT output = new();

            output.PopulateOutput(runResult);

            return output;
        }
        
        public void Dispose()
        {
            SessionOptions.Dispose();
            Session.Dispose();
        }
    }
}