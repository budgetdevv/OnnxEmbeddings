using System;

namespace OnnxEmbeddings.Helpers
{
    public static class TextHelpers
    {
        public static string NormalizedToPercentageNonRounding(this float value, uint decimalPlaces = 2)
        {
            var divideBy = MathF.Pow(10, decimalPlaces);
            
            value *= (divideBy * 100);

            var truncated = (int) value;

            value = truncated / divideBy;
            
            return $"{value}%";
        }
    }
}