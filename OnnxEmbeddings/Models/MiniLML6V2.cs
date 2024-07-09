using FastBertTokenizer;
using Microsoft.ML;
using Microsoft.ML.Data;
using OnnxEmbeddings.Tokenizer;

namespace OnnxEmbeddings.Models
{
    // ReSharper disable once InconsistentNaming
    public sealed class MiniLML6V2
    {
        public class Input(long[] inputIDs, long[] attentionMask)
        {
            // Dimensions: batch, sequence
            [VectorType(1, 256)]
            [ColumnName(INPUT_IDS)]
            public long[] InputIDs { get; } = inputIDs;

            // Dimensions: batch, sequence
            [VectorType(1, 256)]
            [ColumnName(ATTENTION_MASK)]
            public long[] AttentionMask { get; } = attentionMask;
        }
    
        public class Output
        {
            // Dimensions: batch_size, sequence, 384
            // [VectorType(1, 7, 384)]
            [ColumnName(TOKEN_EMBEDDINGS)]
            public float[] TokenEmbeddings { get; set; }
            
            // Dimensions: batch_size, Divsentence_embedding_dim_1
            // [VectorType(1, 384)]
            [ColumnName(SENTENCE_EMBEDDING)]
            public float[] SentenceEmbeddings { get; set; }
        }
        
        public class Configuration
        {
            public const int MAX_SEQUENCE_LENGTH = 256;
            
            public const string MODEL_PATH = "Resources/Models/all-MiniLM-L6-v2.onnx";
        }
        
        private const string 
            INPUT_IDS = "input_ids",
            ATTENTION_MASK = "attention_mask",
            TOKEN_EMBEDDINGS = "token_embeddings",
            SENTENCE_EMBEDDING = "sentence_embedding";

        private static readonly string[] INPUT_COLUMN_NAMES =
        [
            INPUT_IDS,
            ATTENTION_MASK,
        ];
        
        private static readonly string[] OUTPUT_COLUMN_NAMES = 
        [ 
            TOKEN_EMBEDDINGS,
            SENTENCE_EMBEDDING,
        ];

        private readonly Configuration Config;
        private readonly BertTokenizer WordPieceTokenizer;

        public MiniLML6V2(Configuration config)
        {
            Config = config;
            WordPieceTokenizer = Tokenizers.CreateWordPieceTokenizer("sentence-transformers/all-MiniLM-L6-v2").Result;
        }

        public float[] GenerateEmbeddings(string[] sentences, bool meanPooling = true, bool normalize = true)
        {
            var mlContext = new MLContext();
            var batchSize = sentences.Length;
            var encodedCorpus = CreateInput(sentences);
            
            var maxSequenceLength = Configuration.MAX_SEQUENCE_LENGTH;
            
            var encodedCorpusInputIDs = encodedCorpus.InputIDs;
            var encodedCorpusInputIDsLength = encodedCorpusInputIDs.Length;
            
            var encodedCorpusAttentionMask = encodedCorpus.AttentionMask;
            var encodedCorpusAttentionMaskLength = encodedCorpusAttentionMask.Length;

            // Onnx models do not support variable dimension vectors.
            var inputSchema = SchemaDefinition.Create(typeof(Input));

            int[] inputIDsDimensions = [ batchSize, encodedCorpusInputIDsLength ];
            
            inputSchema[INPUT_IDS].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Int64, 
                dimensions: inputIDsDimensions);
            
            int[] attentionMaskDimensions = [ batchSize, encodedCorpusAttentionMaskLength ];
            
            inputSchema[ATTENTION_MASK].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Int64,
                dimensions: attentionMaskDimensions);
            
            // Onnx models may have hardcoded dimensions for inputs. Use a custom
            // schema for variable dimension since the number of text documents
            // are a user input for us (batchSize).
            var inputShape = new Dictionary<string, int[]>
            {
                { INPUT_IDS, inputIDsDimensions },
                { ATTENTION_MASK, attentionMaskDimensions },
            };
            
            var pipeline = mlContext.Transforms
                .ApplyOnnxModel(
                    inputColumnNames: INPUT_COLUMN_NAMES,
                    outputColumnNames: OUTPUT_COLUMN_NAMES,
                    modelFile: Configuration.MODEL_PATH,
                    shapeDictionary: inputShape,
                    gpuDeviceId: null,
                    fallbackToCpu: true);
            
            var trainingData = mlContext.Data.LoadFromEnumerable<Input>([ ], inputSchema);
            var model = pipeline.Fit(trainingData);
            
            // Output schema dimensions: batchSize x sequence x 384
            var outputSchema = SchemaDefinition.Create(typeof(Output));

            int[] tokenEmbeddingsDimensions = [ batchSize, encodedCorpusInputIDsLength, 384 ];
            
            outputSchema[TOKEN_EMBEDDINGS].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Single, 
                dimensions: tokenEmbeddingsDimensions);
            
            int[] sentenceEmbeddingDimensions = [ batchSize, 384 ];
            
            outputSchema[SENTENCE_EMBEDDING].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Single, 
                dimensions: sentenceEmbeddingDimensions);
            
            var engine = mlContext.Model
                .CreatePredictionEngine<Input, Output>(
                    transformer: model,
                    inputSchemaDefinition: inputSchema,
                    outputSchemaDefinition: outputSchema);
            
            var prediction = engine.Predict(encodedCorpus);

            var tokenEmbeddings = prediction.TokenEmbeddings;
            var sentenceEmbeddings = prediction.SentenceEmbeddings;
            
            var tokenEmbeddingsDimensionsLong = tokenEmbeddingsDimensions.Select(x => (long) x).ToArray();
            var attentionMaskDimensionsLong = attentionMaskDimensions.Select(x => (long) x).ToArray();

            // See: https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3#usage-huggingface-transformers
            
            if (!meanPooling)
            {
                if (!normalize)
                {
                    return sentenceEmbeddings;
                }

                else
                {
                    var tokenEmbeddingsTensor = Torch.tensor(sentenceEmbeddings, dimensions: tokenEmbeddingsDimensionsLong);
                    return Normalization.NormalizeTensor(tokenEmbeddingsTensor).data<float>().ToArray();
                }
            }

            else
            {
                var tensor = Pooling.MeanPooling(
                    tokenEmbeddings,
                    encodedCorpusAttentionMask,
                    attentionMaskDimensionsLong,
                    tokenEmbeddingsDimensionsLong);
                
                if (normalize)
                {
                    tensor = Normalization.NormalizeTensor(tensor);
                }

                return tensor.data<float>().ToArray();
            }
        }

        public Input CreateInput(string sentence)
        {
            var (inputIDs, attentionMask, _) = WordPieceTokenizer
                .Encode(input: sentence, maximumTokens: Configuration.MAX_SEQUENCE_LENGTH);
            
            return new(inputIDs.ToArray(), attentionMask.ToArray());
        }

        public Input CreateInput(string[] sentences)
        {
            var tokenizer = WordPieceTokenizer;

            var maxSequenceLength = Configuration.MAX_SEQUENCE_LENGTH;
            
            var batchSize = sentences.Length;
            var bufferSize = batchSize * maxSequenceLength;
            
            // Allocate memory for inputIds and attentionMask
            var inputIds = new long[bufferSize];
            var attentionMask = new long[bufferSize];
            
            // Encode the sentences using the provided method
            tokenizer.Encode(sentences, inputIds, attentionMask, maxSequenceLength);
            
            return new(inputIDs: inputIds, attentionMask: attentionMask);
        }
    }
}