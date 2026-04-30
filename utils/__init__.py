from .textual_utils.data_processing.data_utils import (
    balance_classes,
    build_positive_negative_pairs,
    build_training_dataframe,
    build_vector_text_columns,
    chunk_sequence,
    clean_references,
    clean_citation_dataframe,
    extract_author_names,
    is_not_empty,
    load_citation_dataframe,
    load_citation_dataframe_from_files,
    load_clean_citation_dataframe,
    load_clean_citation_dataframe_from_files,
    load_parquet_files,
    load_parquet_chunks,
    normalize_text,
    resolve_chunk_paths,
)
from .textual_utils.features.feature_extractor import FeatureExtractor, build_classic_ml_matrix
from .embedding_transformer_utils import (
    df_type_from_name,
    find_project_root,
    get_torch_device,
    load_pair_embedding_transformer_model,
    load_simple_transformer_model,
    sample_dataframe,
    set_torch_seed,
    slug,
)

try:
    from .model_classes import (
        BaseModel,
        GraphFeatureTransformer,
        KNNModel,
        PairEmbeddingTransformer,
        PairEmbeddingTransformerModel,
        SimpleTransformer,
        SimpleTransformerModel,
    )
except (ImportError, OSError):
    BaseModel = None
    GraphFeatureTransformer = None
    KNNModel = None
    PairEmbeddingTransformer = None
    PairEmbeddingTransformerModel = None
    SimpleTransformer = None
    SimpleTransformerModel = None

# Optional torch-dependent imports: keep lightweight utilities usable even when
# torch or one of its Windows DLL dependencies is unavailable.
try:
    from .textual_utils.data_processing.citation_dataset import BertCitationDataset, CitationDataset
except (ImportError, OSError):
    BertCitationDataset = None
    CitationDataset = None

try:
    from .textual_utils.models.training import (
        PlotLossCallback,
        WeightedTrainer,
        build_bert_datasets,
        compute_pos_weight,
        create_training_arguments,
        evaluate_predictions,
        get_device,
        get_grade_from_probability,
        incremental_sgd_step,
        predict_trainer_outputs,
        predict_with_grade,
        rolling_accuracy,
        set_seed,
        split_train_test,
        split_dataset,
        build_classification_report,
    )
except (ImportError, OSError):
    PlotLossCallback = None
    WeightedTrainer = None
    build_bert_datasets = None
    compute_pos_weight = None
    create_training_arguments = None
    evaluate_predictions = None
    get_device = None
    get_grade_from_probability = None
    incremental_sgd_step = None
    predict_trainer_outputs = None
    predict_with_grade = None
    rolling_accuracy = None
    set_seed = None
    split_train_test = None
    split_dataset = None
    build_classification_report = None
