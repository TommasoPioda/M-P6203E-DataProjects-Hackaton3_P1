from src.data.data_utils import (
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
from src.features.feature_extractor import FeatureExtractor, build_classic_ml_matrix

# Optional torch-dependent imports: keep lightweight utilities usable even when
# torch or one of its Windows DLL dependencies is unavailable.
try:
    from src.data.citation_dataset import BertCitationDataset, CitationDataset
except (ImportError, OSError):
    BertCitationDataset = None
    CitationDataset = None

try:
    from .training import (
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
