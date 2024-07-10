using OnnxEmbeddings.Helpers;
using OnnxEmbeddings.Models;
using OnnxEmbeddings.Models.HuggingFace;

namespace Sample
{
    internal class Program
    {
        private readonly struct MiniLMConfig: IModelConfig
        {
            public static string ModelPath => "Assets/Models/all-MiniLM-L6-v2.onnx";
        }
        
        private static async Task Main(string[] args)
        {
            var miniLM = await MiniLML6V2<MiniLMConfig>.LoadModelAsync();

            string[] query1 = [ "TrumpMcDonaldz is dumb" ];
            string[] query2 = [ "TrumpMcDonaldz is dumb" ];
            
            var query1Embeddings = miniLM.GenerateEmbeddings(query1, out var query1EmbeddingsDimensions);
            var query2Embeddings = miniLM.GenerateEmbeddings(query2, out var query2EmbeddingsDimensions);
            
            Console.WriteLine($"Query 1 embeddings:\n{GetArrayPrintText(query1Embeddings)}\n");
            Console.WriteLine($"Query 2 embeddings:\n{GetArrayPrintText(query2Embeddings)}\n");
            
            var query1Tensor = Torch.tensor(query1Embeddings, query1EmbeddingsDimensions.ExpandToLong());
            var query2Tensor = Torch.tensor(query2Embeddings, query2EmbeddingsDimensions.ExpandToLong());
            
            var topK = SimilarityHelpers.TopKByCosineSimilarity(
                query1Tensor,
                query2Tensor,
                query1.Length);

            var scores = topK.Values.data<float>();

            var length = topK.Indexes.data<long>().Count;

            for (int i = 0; i < length; i++)
            {
                var currentScore = scores[i];
                Console.WriteLine($"Cosine similarity score: {currentScore.NormalizedToPercentageNonRounding()}\n");
            }
            
            var dotProduct = SimilarityHelpers.DotProduct(query1Embeddings, query2Embeddings);
            Console.WriteLine($"Dot product similarity score: {dotProduct.NormalizedToPercentageNonRounding()}\n");
        }
        
        private static string GetArrayPrintText(ReadOnlySpan<float> arr)
        {
            var text = "[";
            
            foreach (var item in arr)
            {
                text += $" {item},";
            }
            
            var length = text.Length;
            
            if (length > 1)
            {
                text = text.Remove(length - 1);
            }

            text += " ]";

            return text;
        }
    }
}