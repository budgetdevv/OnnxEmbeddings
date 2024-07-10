using System.Threading.Tasks;
using FastBertTokenizer;

namespace OnnxEmbeddings.Tokenizer
{
    public static class Tokenizers
    {
        public static async ValueTask<BertTokenizer> CreateWordPieceTokenizer(string huggingFaceRepoURL)
        {
            var tokenizer = new BertTokenizer();
            await tokenizer.LoadFromHuggingFaceAsync(huggingFaceRepoURL);
            return tokenizer;
        }
    }
}