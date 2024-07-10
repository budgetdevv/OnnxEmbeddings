namespace OnnxEmbeddings.Helpers
{
    public static class SimilarityHelpers
    {
        public static (TorchTensor Values, TorchTensor Indexes) TopKByCosineSimilarity(
            TorchTensor corpus,
            TorchTensor query,
            int limit)
        {
            // Cosine similarity of two tensors of different dimensions.
            // cos_sim(a, b) = dot_product(a_norm, transpose(b_norm))
            // a_norm and b_norm are L2 norms of the tensors.
            var corpusNorm = corpus / corpus.norm(1).unsqueeze(-1);
            var queryNorm = query / query.norm(1).unsqueeze(-1);
            var similar = queryNorm.mm(corpusNorm.transpose(0, 1));

            // Compute top K values in the similarity result and return the
            // values and indexes of the elements.
            return similar.topk(limit);
        }

        public static float DotProduct(float[] array1, float[] array2) 
        {
            var result = 0f;
            
            for (int i = 0; i < array1.Length; i++)
            {
                result += array1[i] * array2[i];
            }

            return result;
        }
    }
}