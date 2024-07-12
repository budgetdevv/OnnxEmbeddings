using System;

namespace OnnxEmbeddings.Helpers
{
    public static class PoolingHelpers
    {
        public static TorchTensor MeanPooling(
            float[] tokenEmbeddings,
            long[] attentionMask,
            ReadOnlySpan<long> attentionMaskDimensions,
            ReadOnlySpan<long> tokenEmbeddingsDimensions)
        {
            var tokenEmbeddingsTensor = Torch.tensor(rawArray: tokenEmbeddings, dimensions: tokenEmbeddingsDimensions);

            var attentionMaskExpandedTensor = Torch
                .tensor(dataArray: attentionMask, dimensions: attentionMaskDimensions)
                .unsqueeze(dim: -1)
                .expand(sizes: tokenEmbeddingsTensor.shape)
                .to(Torch.float32);

            var sumEmbeddings = (tokenEmbeddingsTensor * attentionMaskExpandedTensor).sum(1);

            var sumMask = attentionMaskExpandedTensor
                .sum(dim: 1)
                .clamp(min: 1e-9, max: float.MaxValue);

            return sumEmbeddings / sumMask;
        }
    }
}