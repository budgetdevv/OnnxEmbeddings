namespace OnnxEmbeddings.Models
{
    public static class Pooling
    {
        private static readonly long[] ONE = [ 1L ];
        
        public static TorchTensor MeanPooling(
            float[] tokenEmbeddings, 
            long[] attentionMask,
            long[] attentionMaskDimensions,
            long[] tokenEmbeddingsDimensions)
        {
            var tokenEmbeddingsTensor = Torch.tensor(rawArray: tokenEmbeddings, dimensions: tokenEmbeddingsDimensions);
                
            var attentionMaskExpanded = Torch
                .tensor(dataArray: attentionMask, dimensions: attentionMaskDimensions)
                .unsqueeze(-1)
                .expand(tokenEmbeddingsTensor.shape)
                .to(Torch.float32);
            
            var sumEmbeddings = (tokenEmbeddingsTensor * attentionMaskExpanded).sum(ONE);
            var sumMask = attentionMaskExpanded.sum(ONE).clamp(1e-9, float.MaxValue);

            return sumEmbeddings / sumMask;
        }
    }
}