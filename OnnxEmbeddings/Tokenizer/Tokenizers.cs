using FastBertTokenizer;

namespace OnnxEmbeddings.Tokenizer
{
    public static class Tokenizers
    {
        public static async ValueTask<BertTokenizer> CreateWordPieceTokenizer(string huggingFaceURL)
        {
            var tokenizer = new BertTokenizer();
            await tokenizer.LoadFromHuggingFaceAsync(huggingFaceURL);
            return tokenizer;
        }
    }
}