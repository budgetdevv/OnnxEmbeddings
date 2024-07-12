using System;
using System.Threading.Tasks;
using FastBertTokenizer;
using OnnxEmbeddings.Helpers;
using OnnxEmbeddings.Tokenizer;

namespace OnnxEmbeddings.Models.HuggingFace
{
    public sealed class GTELargeENV1_5<ConfigT>: IHuggingFaceModel<GTELargeENV1_5<ConfigT>, ConfigT>
        where ConfigT : struct, IModelConfig
    {
        public static int MAX_SEQUENCE_LENGTH => 8192;
        public static int EMBEDDING_DIMENSION => 1024;
        
        public static string HuggingFaceRepoName => "Alibaba-NLP/gte-large-en-v1.5";

        private readonly BertTokenizer WordPieceTokenizer;

        private readonly SentenceEmbedder<SentenceEmbedder.InputExtended, SentenceEmbedder.LastHiddenStateOutput> Embedder;
        
        private GTELargeENV1_5(BertTokenizer wordPieceTokenizer)
        {
            WordPieceTokenizer = wordPieceTokenizer;
            Embedder = new(ConfigT.ModelPath);
        }

        public static async ValueTask<GTELargeENV1_5<ConfigT>> LoadModelAsync()
        {
            return new(await Tokenizers.CreateWordPieceTokenizer(HuggingFaceRepoName));
        }

        public float[] GenerateEmbeddings(string[] sentences, int maxSequenceLength, out int[] outputDimensions)
        {
            return GenerateEmbeddings(sentences, maxSequenceLength, normalize: true, out outputDimensions);
        }

        public float[] GenerateEmbeddings(string[] sentences, int maxSequenceLength, bool normalize, out int[] outputDimensions)
        {
            if (maxSequenceLength > MAX_SEQUENCE_LENGTH)
            {
                throw new ArgumentException(
                    $"The provided max sequence length {maxSequenceLength} is greater than the maximum supported sequence length {MAX_SEQUENCE_LENGTH}.",
                    nameof(maxSequenceLength));
            }
            
            var batchSize = sentences.Length;
            var input = new SentenceEmbedder.InputExtended(sentences, maxSequenceLength, WordPieceTokenizer);

            var lastHiddenState = Embedder.GenerateEmbeddings(input).LastHiddenState;
            
            var embeddings = new float[batchSize * EMBEDDING_DIMENSION];

            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                for (int dimensionIndex = 0; dimensionIndex < EMBEDDING_DIMENSION; dimensionIndex++)
                {
                    embeddings[batchIndex * EMBEDDING_DIMENSION + dimensionIndex] = lastHiddenState[batchIndex * maxSequenceLength * EMBEDDING_DIMENSION + dimensionIndex];
                }
            }

            outputDimensions = [ batchSize, EMBEDDING_DIMENSION ];
                
            if (!normalize)
            {
                return embeddings;
            }
            
            else
            {
                var tokenEmbeddingsTensor = Torch.tensor(embeddings, dimensions: outputDimensions.ExpandToLong());
                return TorchHelpers.NormalizeTensor(tokenEmbeddingsTensor, dim: 1).data<float>().ToArray();
            }
        }

        public void Dispose()
        {
            Embedder.Dispose();
        }
    }
}
