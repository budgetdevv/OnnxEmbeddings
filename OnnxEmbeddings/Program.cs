using System;
using OnnxEmbeddings.Helpers;
using OnnxEmbeddings.Models;

namespace OnnxEmbeddings
{
    internal class Program
    {
        private static void PrintArr(ReadOnlySpan<float> arr)
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
            
            Console.WriteLine(text);
        }
        
        private static void Main(string[] args)
        {
            var miniLM = new MiniLML6V2(new());

            string[] query1 = [ "That is a happy person" ];
            string[] query2 = [ "That is a happy person" ];
            
            
            var query1Embeddings = miniLM.GenerateEmbeddings(query1, out var query1EmbeddingsDimensions);
            var query2Embeddings = miniLM.GenerateEmbeddings(query2, out var query2EmbeddingsDimensions);
            
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
            Console.WriteLine($"Dot product similarity score: {dotProduct.NormalizedToPercentageNonRounding()}");
        }
    }
}