using System;
using System.Linq;
using System.Threading.Tasks;
using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxEmbeddings.Helpers;
using OnnxEmbeddings.Tokenizer;

namespace OnnxEmbeddings.Models.HuggingFace
{
    // ReSharper disable once InconsistentNaming
    public sealed class MiniLML6V2<ConfigT>: IHuggingFaceModel<MiniLML6V2<ConfigT>, ConfigT> 
        where ConfigT: struct, IModelConfig
    {
        private struct Output: SentenceEmbedder.ISentenceEmbedderOutput
        {
            public const string 
                TOKEN_EMBEDDINGS = "token_embeddings",
                SENTENCE_EMBEDDING = "sentence_embedding";
            
            public static string[] OutputNames => [ TOKEN_EMBEDDINGS, SENTENCE_EMBEDDING ];
            
            // Dimensions: batch_size, sequence, 384
            public float[] TokenEmbeddings { get; set; }
            
            // Dimensions: batch_size, Divsentence_embedding_dim_1
            public float[] SentenceEmbeddings { get; set; }
            
            public void PopulateOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output)
            {
                foreach (var value in output)
                {
                    var arr = ((DenseTensor<float>) value.Value).ToArray();
                    
                    switch (value.Name)
                    {
                        case TOKEN_EMBEDDINGS:
                            TokenEmbeddings = arr;
                            break;
                        
                        case SENTENCE_EMBEDDING:
                            SentenceEmbeddings = arr;
                            break;
                    }
                }
            }
        }
        
        public static int MAX_SEQUENCE_LENGTH => 256;
        public static int EMBEDDING_DIMENSION => 384;

        public static string HuggingFaceRepoName => "sentence-transformers/all-MiniLM-L6-v2";
        
        private readonly BertTokenizer WordPieceTokenizer;
        
        private readonly SentenceEmbedder<SentenceEmbedder.Input, Output> Embedder;

        private MiniLML6V2(BertTokenizer wordPieceTokenizer)
        {
            WordPieceTokenizer = wordPieceTokenizer;
            Embedder = new(ConfigT.ModelPath);
        }
        
        public static async ValueTask<MiniLML6V2<ConfigT>> LoadModelAsync()
        {
            return new(await Tokenizers.CreateWordPieceTokenizer(HuggingFaceRepoName));
        }
        
        public float[] GenerateEmbeddings(string[] sentences, int maxSequenceLength, out int[] outputDimensions)
        {
            return GenerateEmbeddings(
                sentences: sentences,
                maxSequenceLength: maxSequenceLength,
                meanPooling: true,
                normalize: true,
                sentenceEmbeddingDimensions: out outputDimensions);
        }

        public float[] GenerateEmbeddings(
            string[] sentences,
            int maxSequenceLength,
            bool meanPooling,
            bool normalize,
            out int[] sentenceEmbeddingDimensions)
        {
            if (maxSequenceLength > MAX_SEQUENCE_LENGTH)
            {
                throw new ArgumentException(
                    $"The provided max sequence length {maxSequenceLength} is greater than the maximum supported sequence length {MAX_SEQUENCE_LENGTH}.",
                    nameof(maxSequenceLength));
            }
            
            var batchSize = sentences.Length;
            
            var input = new SentenceEmbedder.Input(sentences, maxSequenceLength, WordPieceTokenizer);
            
            var attentionMask = input.AttentionMask;
            
            var attentionMaskDimensionsLong = (ReadOnlySpan<long>) input.Dimensions
                .ExpandToLong()
                .AsSpan();

            var output = Embedder.GenerateEmbeddings(input);

            var tokenEmbeddings = output.TokenEmbeddings;
            var sentenceEmbeddings = output.SentenceEmbeddings;

            int[] tokenEmbeddingsDimensions = [ batchSize, maxSequenceLength, EMBEDDING_DIMENSION ];
            
            var tokenEmbeddingsDimensionsLong = (ReadOnlySpan<long>) tokenEmbeddingsDimensions
                .ExpandToLong()
                .AsSpan();
            
            sentenceEmbeddingDimensions = [ batchSize, EMBEDDING_DIMENSION ];

            // See: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            
            if (meanPooling)
            {
                var tensor = PoolingHelpers.MeanPooling(
                    tokenEmbeddings,
                    attentionMask,
                    attentionMaskDimensionsLong,
                    tokenEmbeddingsDimensionsLong);
                
                if (normalize)
                {
                    tensor = TorchHelpers.NormalizeTensor(tensor);
                }

                return tensor.data<float>().ToArray();
            }
            
            else
            {
                if (!normalize)
                {
                    return sentenceEmbeddings;
                }

                else
                {
                    var tokenEmbeddingsTensor = Torch.tensor(sentenceEmbeddings, dimensions: tokenEmbeddingsDimensionsLong);
                    return TorchHelpers.NormalizeTensor(tokenEmbeddingsTensor).data<float>().ToArray();
                }
            }
        }
        
        public void Dispose()
        {
            Embedder.Dispose();
        }
    }
}