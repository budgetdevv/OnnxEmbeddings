namespace OnnxEmbeddings.Models
{
    public static class Normalization
    {
        public static TorchTensor NormalizeTensor(TorchTensor input, float p = 2f, int dim = -1, bool keep = true, float eps = 1e-12f)
        {
            var denom = input.norm(dim, keep, p).clamp_min(eps);

            if (keep)
            {
                denom = denom.expand((ReadOnlySpan<long>) [-1, -1]);
            }
            

            return input / denom;
        }
    }
}