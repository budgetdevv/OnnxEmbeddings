namespace OnnxEmbeddings.Models.HuggingFace
{
    public interface IHuggingFaceModel<ModelT, ConfigT>: IModel<ModelT, ConfigT>
        where ModelT: class, IHuggingFaceModel<ModelT, ConfigT>
        where ConfigT: struct, IModelConfig
    {
        public static abstract string HuggingFaceRepoName { get; }
    }
}