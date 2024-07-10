using System.Collections.Generic;
using FastBertTokenizer;
using Microsoft.ML;
using Microsoft.ML.Data;
using OnnxEmbeddings.Helpers;
using OnnxEmbeddings.Tokenizer;

namespace OnnxEmbeddings.Models
{
    // ReSharper disable once InconsistentNaming
    public sealed class MiniLML6V2
    {
        private class Input(long[] inputIDs, long[] attentionMask)
        {
            // Dimensions: batch, sequence
            // ReSharper disable once UnusedMember.Local ( It is used by ML.NET )
            [ColumnName(INPUT_IDS)]
            public long[] InputIDs { get; } = inputIDs;

            // Dimensions: batch, sequence
            [ColumnName(ATTENTION_MASK)]
            public long[] AttentionMask { get; } = attentionMask;
        }
    
        private sealed class Output
        {
            // Dimensions: batch_size, sequence, 384
            // ReSharper disable once UnusedAutoPropertyAccessor.Local ( It is used by ML.NET )
            [ColumnName(TOKEN_EMBEDDINGS)]
            public float[] TokenEmbeddings { get; set; }
            
            // Dimensions: batch_size, Divsentence_embedding_dim_1
            // ReSharper disable once UnusedAutoPropertyAccessor.Local ( It is used by ML.NET )
            [ColumnName(SENTENCE_EMBEDDING)]
            public float[] SentenceEmbeddings { get; set; }
        }
        
        public class Configuration(string modelPath)
        {
            public readonly string ModelPath = modelPath;
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
        
        private const int 
            MAX_SEQUENCE_LENGTH = 256,
            EMBEDDING_DIMENSION = 384;

        public MiniLML6V2(Configuration config)
        {
            Config = config;
            WordPieceTokenizer = Tokenizers.CreateWordPieceTokenizer("sentence-transformers/all-MiniLM-L6-v2").Result;
        }

        public float[] GenerateEmbeddings(string[] sentences, out int[] sentenceEmbeddingDimensions)
        {
            return GenerateEmbeddings(
                sentences: sentences,
                meanPooling: true,
                normalize: true,
                sentenceEmbeddingDimensions: out sentenceEmbeddingDimensions);
        }

        public float[] GenerateEmbeddings(
            string[] sentences, 
            bool meanPooling,
            bool normalize,
            out int[] sentenceEmbeddingDimensions)
        {
            var mlContext = new MLContext();
            
            var batchSize = sentences.Length;
            
            var encodedCorpus = CreateInput(sentences);
            
            var maxSequenceLength = MAX_SEQUENCE_LENGTH;
            
            var encodedCorpusAttentionMask = encodedCorpus.AttentionMask;

            // Onnx models do not support variable dimension vectors.
            var inputSchema = SchemaDefinition.Create(typeof(Input));

            int[] inputIDsDimensions = [ batchSize, maxSequenceLength ];
            
            inputSchema[INPUT_IDS].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Int64, 
                dimensions: inputIDsDimensions);
            
            int[] attentionMaskDimensions = [ batchSize, maxSequenceLength ];
            
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
                    modelFile: Config.ModelPath,
                    shapeDictionary: inputShape,
                    gpuDeviceId: null,
                    fallbackToCpu: true);
            
            var trainingData = mlContext.Data.LoadFromEnumerable<Input>([ ], inputSchema);
            var model = pipeline.Fit(trainingData);
            
            var outputSchema = SchemaDefinition.Create(typeof(Output));

            int[] tokenEmbeddingsDimensions = [ batchSize, maxSequenceLength, EMBEDDING_DIMENSION ];
            
            outputSchema[TOKEN_EMBEDDINGS].ColumnType = new VectorDataViewType(
                itemType: NumberDataViewType.Single, 
                dimensions: tokenEmbeddingsDimensions);
            
            sentenceEmbeddingDimensions = [ batchSize, EMBEDDING_DIMENSION ];
            
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

            var tokenEmbeddingsDimensionsLong = tokenEmbeddingsDimensions.ExpandToLong();
            var attentionMaskDimensionsLong = attentionMaskDimensions.ExpandToLong();

            // See: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            
            if (meanPooling)
            {
                var tensor = PoolingHelpers.MeanPooling(
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
            
            else
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
        }

        private Input CreateInput(string sentence)
        {
            var (inputIDs, attentionMask, _) = WordPieceTokenizer
                .Encode(input: sentence, maximumTokens: MAX_SEQUENCE_LENGTH);
            
            return new(inputIDs.ToArray(), attentionMask.ToArray());
        }

        private Input CreateInput(string[] sentences)
        {
            var batchSize = sentences.Length;
            var bufferSize = batchSize * MAX_SEQUENCE_LENGTH;
            
            // Allocate memory for inputIds and attentionMask
            var inputIDs = new long[bufferSize];
            var attentionMask = new long[bufferSize];
            
            // Encode the sentences using the provided method
            WordPieceTokenizer.Encode(sentences, inputIDs, attentionMask, MAX_SEQUENCE_LENGTH);
            
            return new(inputIDs: inputIDs, attentionMask: attentionMask);
        }
    }
}