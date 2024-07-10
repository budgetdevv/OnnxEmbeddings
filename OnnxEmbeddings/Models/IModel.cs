using System;
using System.Threading.Tasks;

namespace OnnxEmbeddings.Models
{
    public interface IModel<ModelT, ModelConfigT>: IDisposable
        where ModelT: class, IModel<ModelT, ModelConfigT>
        where ModelConfigT: struct, IModelConfig
    {
        public static abstract ValueTask<ModelT> LoadModelAsync();
    }

    public interface IModelConfig
    {
        public static abstract string ModelPath { get; }
    }
}