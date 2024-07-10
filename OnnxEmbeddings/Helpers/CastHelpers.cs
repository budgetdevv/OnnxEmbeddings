using System;
using System.Numerics;

namespace OnnxEmbeddings.Helpers
{
    public static class CastHelpers
    {
        public static long[] ExpandToLong(this int[] inputArr)
        {
            return inputArr.AsSpan().ExpandToLong();
        }
        
        public static long[] ExpandToLong(this Span<int> inputSpan)
        {
            // TODO: Speed this up further with explicit platform intrinsics and unsafe code.

            var length = inputSpan.Length;

            var intVectorWidth = Vector<int>.Count;
            var longVectorWidth = Vector<long>.Count;

            var result = new long[length];
            var resultSpan = result.AsSpan();

            var pow2Length = length & ~(intVectorWidth - 1);

            for (int i = 0; i < pow2Length; i += intVectorWidth)
            {
                var sourceSpan = inputSpan.Slice(i, intVectorWidth);
                var destUpperHalfSpan = resultSpan.Slice(i, longVectorWidth);
                var destLowerHalfSpan = resultSpan.Slice(i + longVectorWidth, longVectorWidth);

                var vector = new Vector<int>(sourceSpan);

                Vector.Widen(vector, out var upper, out var lower);

                upper.CopyTo(destUpperHalfSpan);
                lower.CopyTo(destLowerHalfSpan);
            }

            var remainder = length - pow2Length;

            if (remainder != 0)
            {
                var sourceSpan = inputSpan.Slice(pow2Length, remainder);
                var destSpan = result.AsSpan(pow2Length, remainder);

                for (int i = 0; i < remainder; i++)
                {
                    destSpan[i] = sourceSpan[i];
                }
            }

            return result;
        }
    }
}