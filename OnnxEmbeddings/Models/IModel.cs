using System;
using System.Threading.Tasks;

namespace OnnxEmbeddings.Models
{
    public interface IModel<ModelT, ModelConfigT>: IDisposable
        where ModelT: class, IModel<ModelT, ModelConfigT>
        where ModelConfigT: struct, IModelConfig
    {
        public static abstract int MAX_SEQUENCE_LENGTH { get; }
        
        public static abstract int EMBEDDING_DIMENSION { get; }
        
        public static abstract ValueTask<ModelT> LoadModelAsync();
        public float[] GenerateEmbeddings(string[] sentences, int maxSequenceLength, out int[] outputDimensions);
    }

    public interface IModelConfig
    {
        public static abstract string ModelPath { get; }
    }
}