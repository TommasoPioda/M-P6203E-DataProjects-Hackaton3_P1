import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Keep prediction memory use predictable.
N_JOBS = 1

def load_latest_models(base_path='./Models'):
    """
    Creates a nested dictionary registry.
    Access via: registry['folder_name']['KNN']
    """
    models_registry = {}
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Error: Directory {base_path} not exist.")
        return models_registry

    model_types = {
        'KNN': 'Best_KNN',
        'XGB': 'Best_XGB',
        'transformer': 'Transformer'
    }

    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            models_registry[subdir.name] = {}
            for label, keyword in model_types.items():
                model_files = [f for f in subdir.glob(f'*{keyword}*.pkl')]
                if not model_files:
                    continue

                latest_file = sorted(model_files, key=lambda x: x.name)[-1]
                try:
                    with open(latest_file, 'rb') as f:
                        model = pickle.load(f)
                    if hasattr(model, 'set_params') and 'n_jobs' in model.get_params():
                        model.set_params(n_jobs=N_JOBS)
                    models_registry[subdir.name][label] = model
                    print(f"Loaded {label} from {subdir.name}: {latest_file.name}")
                except Exception as e:
                    print(f"Failed to load {label} in {subdir.name}: {e}")

    return models_registry



def _as_flat_model_input(X):
    """Return a numeric 2D array for sklearn/XGBoost-style models."""
    if hasattr(X, "to_numpy"):
        return X.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _pair_embedding_dim(model):
    """Return expected per-embedding width for pair transformers, if detectable."""
    torch_model = getattr(model, "model", None)
    input_projection = getattr(torch_model, "input_projection", None)
    return getattr(input_projection, "in_features", None)


def _as_pair_transformer_input(model, X):
    """Reshape flat pair embeddings from (n, 2*d) to (n, 2, d)."""
    embedding_dim = _pair_embedding_dim(model)
    if embedding_dim is None:
        return None

    X_array = _as_flat_model_input(X)
    if X_array.ndim != 2 or X_array.shape[1] % embedding_dim != 0:
        return None

    n_parts = X_array.shape[1] // embedding_dim
    if n_parts < 2:
        return None

    return X_array.reshape(X_array.shape[0], n_parts, embedding_dim)


def _predict_safely(model, X):
    pair_input = _as_pair_transformer_input(model, X)
    if pair_input is not None:
        try:
            return model.predict(pair_input)
        except (ValueError, RuntimeError):
            pass

    try:
        return model.predict(X)
    except (ValueError, RuntimeError) as exc:
        flat_input = _as_flat_model_input(X)
        try:
            return model.predict(flat_input)
        except (ValueError, RuntimeError):
            if pair_input is not None:
                return model.predict(pair_input)
            raise exc


def evaluate_model_on_sets(model, set_dict):
    """
    Evaluates a single model on its specific dictionary of sets.
    Returns a dict with confusion matrices and a dataframe of metrics.
    """
    results = {'cms': {}, 'metrics': []}

    for set_name, (X, y) in set_dict.items():
        y_pred = _predict_safely(model, X)
        results['cms'][set_name] = confusion_matrix(y, y_pred)
        results['metrics'].append({
            'Set': set_name,
            'F1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y, y_pred, average='weighted', zero_division=0)
        })
        del y_pred

    results['metrics_df'] = pd.DataFrame(results['metrics']).set_index('Set')
    return results


def plot_model_comparison(model_list, model_names, all_sets_list, title="Model Comparison", figsize=(18, 10)):
    """
    Plot confusion matrices and basic metrics. Returns (fig, metrics_df).
    """
    if not model_list:
        raise ValueError("model_list is empty: nothing to plot.")

    n_models = len(model_list)
    set_names = list(all_sets_list[0].keys())
    n_sets = len(set_names)

    fig, axes = plt.subplots(n_sets, n_models + 1, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).reshape(n_sets, n_models + 1)
    fig.suptitle(title, fontsize=20, fontweight='bold')

    all_evaluations = []
    metrics_rows = []

    for i, (model, m_name, s_dict) in enumerate(zip(model_list, model_names, all_sets_list)):
        res = evaluate_model_on_sets(model, s_dict)
        all_evaluations.append(res)

        for set_name, row in res['metrics_df'].iterrows():
            row_dict = row.to_dict()
            row_dict['Model_Type'] = m_name
            row_dict['Set'] = set_name
            metrics_rows.append(row_dict)

        for j, s_name in enumerate(set_names):
            ax = axes[j, i]
            sns.heatmap(res['cms'][s_name], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{m_name} - {s_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

    for j, s_name in enumerate(set_names):
        ax = axes[j, -1]
        stats_data = []
        for idx, m_name in enumerate(model_names):
            m_metrics = all_evaluations[idx]['metrics_df'].loc[s_name]
            for metric_name in ['F1', 'Accuracy']:
                stats_data.append({'Model': m_name, 'Metric': metric_name, 'Score': m_metrics[metric_name]})

        df_plot = pd.DataFrame(stats_data)
        sns.barplot(data=df_plot, x='Score', y='Metric', hue='Model', ax=ax)
        ax.set_xlim(0, 1.0)
        ax.set_title(f"Global Stats - {s_name}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    metrics_df = pd.DataFrame(metrics_rows).set_index(['Model_Type', 'Set'])
    return fig, metrics_df


def save_show_close(fig, filename, path_plots, save=True):
    output_path = path_plots / filename
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure: {output_path}")
    display(fig)
    plt.close(fig)
    return output_path


def separe_dataset(dataset, target='is_reference_valid', id_columns=['article_id', 'ref_id']):
    columns_to_drop = [target, "split", *id_columns]
    X = dataset.drop(columns=columns_to_drop, errors="ignore")
    y = dataset[target]
    return X, y


def set_dict(datasets, names=['train', 'test'], target='is_reference_valid', id_columns=['article_id', 'ref_id']):
    res_dict = {}
    for df, name in zip(datasets, names):
        X, y = separe_dataset(df, target=target, id_columns=id_columns)
        res_dict[name] = (X, y)
    return res_dict
