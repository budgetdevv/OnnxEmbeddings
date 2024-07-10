﻿using System.Collections.Generic;
using System.Threading.Tasks;
using FastBertTokenizer;
using Microsoft.ML;
using Microsoft.ML.Data;
using OnnxEmbeddings.Helpers;
using OnnxEmbeddings.Tokenizer;

namespace OnnxEmbeddings.Models.HuggingFace
{
    // ReSharper disable once InconsistentNaming
    public sealed class GTELargeENV1_5<ConfigT>: IHuggingFaceModel<GTELargeENV1_5<ConfigT>, ConfigT> 
        where ConfigT: struct, IModelConfig
    {
        private class Input(long[] inputIDs, long[] attentionMask, long[] tokenTypeIDs)
        {
            public const string
                INPUT_IDS = "input_ids",
                ATTENTION_MASK = "attention_mask",
                TOKEN_TYPE_IDS = "token_type_ids";
            
            // Dimensions: batch, sequence
            // ReSharper disable once UnusedMember.Local ( It is used by ML.NET )
            [ColumnName(INPUT_IDS)]
            public long[] InputIDs { get; } = inputIDs;

            // Dimensions: batch, sequence
            [ColumnName(ATTENTION_MASK)]
            public long[] AttentionMask { get; } = attentionMask;
            
            [ColumnName(TOKEN_TYPE_IDS)]
            public long[] TokenTypeIDs { get; } = tokenTypeIDs;
        }
    
        private sealed class Output
        {
            public const string LAST_HIDDEN_STATE = "last_hidden_state";
            
            // Dimensions: batch_size, sequence, 1024
            // ReSharper disable once UnusedAutoPropertyAccessor.Local ( It is used by ML.NET )
            [ColumnName(LAST_HIDDEN_STATE)]
            public float[] LastHiddenState { get; set; }
        }

        private static readonly string[] INPUT_COLUMN_NAMES =
        [
            Input.INPUT_IDS,
            Input.ATTENTION_MASK,
            Input.TOKEN_TYPE_IDS,
        ];
        
        private static readonly string[] OUTPUT_COLUMN_NAMES = 
        [ 
            Output.LAST_HIDDEN_STATE,
        ];

        public static int MAX_SEQUENCE_LENGTH => 8192;
        public static int EMBEDDING_DIMENSION => 1024;

        public static string HuggingFaceRepoName => "Alibaba-NLP/gte-large-en-v1.5";
        
        private readonly BertTokenizer WordPieceTokenizer;

        private GTELargeENV1_5(BertTokenizer wordPieceTokenizer)
        {
            WordPieceTokenizer = wordPieceTokenizer;
        }
        
        public static async ValueTask<GTELargeENV1_5<ConfigT>> LoadModelAsync()
        {
            return new(await Tokenizers.CreateWordPieceTokenizer(HuggingFaceRepoName));
        }
        
        public float[] GenerateEmbeddings(string[] sentences, out int[] outputDimensions)
        {
            return GenerateEmbeddings(
                sentences: sentences,
                meanPooling: true,
                normalize: true,
                outputDimensions: out outputDimensions);
        }

        public float[] GenerateEmbeddings(
            string[] sentences, 
            bool meanPooling,
            bool normalize,
            out int[] outputDimensions)
        {
            var mlContext = new MLContext();
            
            var batchSize = sentences.Length;
            
            var encodedCorpus = CreateInput(sentences);
            
            var maxSequenceLength = MAX_SEQUENCE_LENGTH;
            
            var encodedCorpusAttentionMask = encodedCorpus.AttentionMask;

            // Onnx models do not support variable dimension vectors.
            var inputSchema = SchemaDefinition.Create(typeof(Input));

            int[] inputIDsDimensions = [ batchSize, maxSequenceLength ];
            
            inputSchema[Input.INPUT_IDS].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Int64, 
                dimensions: inputIDsDimensions);

            var attentionMaskDimensions = inputIDsDimensions;
            
            inputSchema[Input.ATTENTION_MASK].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Int64,
                dimensions: attentionMaskDimensions);

            var tokenTypeIDsDimensions = inputIDsDimensions;
            
            inputSchema[Input.TOKEN_TYPE_IDS].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Int64,
                dimensions: tokenTypeIDsDimensions);
            
            // Onnx models may have hardcoded dimensions for inputs. Use a custom
            // schema for variable dimension since the number of text documents
            // are a user input for us (batchSize).
            var inputShape = new Dictionary<string, int[]>
            {
                { Input.INPUT_IDS, inputIDsDimensions },
                { Input.ATTENTION_MASK, attentionMaskDimensions },
                { Input.TOKEN_TYPE_IDS, tokenTypeIDsDimensions },
            };
            
            var pipeline = mlContext.Transforms
                .ApplyOnnxModel(
                    inputColumnNames: INPUT_COLUMN_NAMES,
                    outputColumnNames: OUTPUT_COLUMN_NAMES,
                    modelFile: ConfigT.ModelPath,
                    shapeDictionary: inputShape,
                    gpuDeviceId: null,
                    fallbackToCpu: true);
            
            var trainingData = mlContext.Data.LoadFromEnumerable<Input>([ ], inputSchema);
            var model = pipeline.Fit(trainingData);
            
            var outputSchema = SchemaDefinition.Create(typeof(Output));

            var lastHiddenStateDimensions = outputDimensions = [ batchSize, maxSequenceLength, EMBEDDING_DIMENSION ];
            
            outputSchema[Output.LAST_HIDDEN_STATE].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Single, 
                dimensions: lastHiddenStateDimensions);
            
            var engine = mlContext.Model
                .CreatePredictionEngine<Input, Output>(
                    transformer: model,
                    inputSchemaDefinition: inputSchema,
                    outputSchemaDefinition: outputSchema);
            
            var prediction = engine.Predict(encodedCorpus);

            var lastHiddenState = prediction.LastHiddenState;

            var lastHiddenStateDimensionsLong = lastHiddenStateDimensions.ExpandToLong();
            
            // See: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            
            if (!normalize)
            {
                return lastHiddenState;
            }

            else
            {
                var tokenEmbeddingsTensor = Torch.tensor(lastHiddenState, dimensions: lastHiddenStateDimensionsLong);
                return TorchHelpers.NormalizeTensor(tokenEmbeddingsTensor, dim: 1).data<float>().ToArray();
            }
        }

        private Input CreateInput(string[] sentences)
        {
            var batchSize = sentences.Length;
            var bufferSize = batchSize * MAX_SEQUENCE_LENGTH;
            
            // Allocate memory for inputIds and attentionMask
            var inputIDs = new long[bufferSize];
            var attentionMask = new long[bufferSize];
            var tokenTypeIDs = new long[bufferSize];
            
            // Encode the sentences using the provided method
            WordPieceTokenizer.Encode(sentences, inputIDs, attentionMask, tokenTypeIDs, MAX_SEQUENCE_LENGTH);
            
            return new(inputIDs: inputIDs, attentionMask: attentionMask, tokenTypeIDs: tokenTypeIDs);
        }
        
        public void Dispose()
        {
            // TODO release managed resources here
        }
    }
}