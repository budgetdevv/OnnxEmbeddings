using System;

namespace OnnxEmbeddings.Helpers
{
    public static class TorchHelpers
    {
        // This is from PyTorch's `torch.nn.functional.normalize` function.
        public static TorchTensor NormalizeTensor(TorchTensor input, float p = 2.0f, int dim = 1, float eps = 1e-12f, TorchTensor? output = null)
        {
            var denom = input.norm(p: p, dim: dim, keepdim: true).clamp_min(eps).expand_as(input);
            return input / denom;
        }
    }
}