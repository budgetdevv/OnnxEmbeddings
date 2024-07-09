using OnnxEmbeddings.Models;

namespace OnnxEmbeddings
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var miniLM = new MiniLML6V2(new());

            string[] query1 = [ "That is a happy person" ];
            string[] query2 = [ "That is a happy person" ];
            
            
            var query1Embeddings = miniLM.GenerateEmbeddings(query1);
            var query2Embeddings = miniLM.GenerateEmbeddings(query2);

            ReadOnlySpan<long> dimensions = [1, query1Embeddings.Length];
            
            var query1Tensor = Torch.tensor(query1Embeddings, dimensions);
            var query2Tensor = Torch.tensor(query2Embeddings, dimensions);
            
            var topK = Similarity.TopKByCosineSimilarity(
                query1Tensor,
                query2Tensor,
                query1.Length);

            using var scores = topK.Values.data<float>().GetEnumerator();
            
            foreach (var index in topK.Indexes.data<long>().ToArray())
            {
                scores.MoveNext();
                Console.WriteLine($"Cosine similarity score: {scores.Current*100:f12}");
                Console.WriteLine();
            }
            
            var dotP = Similarity.DotProduct(query1Embeddings, query2Embeddings);
            Console.WriteLine($"Dot product similarity score: {dotP * 100:f12}");
        }
    }
}