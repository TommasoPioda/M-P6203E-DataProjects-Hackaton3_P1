from utils.textual_utils.registry.legacy_model_saver import save_model_artifact

__all__ = ["save_model_artifact"]


def save_model_artifact(model, df_name = None, model_name = None, relative_model_dir = None):
    """
    Save the model artifact using the legacy model saver.

    Args:
        model: The model to be saved.
        df_name: The name of the dataframe.
        model_name: The name of the model.
        relative_model_dir: The relative directory where the model will be saved.
    """
    import pickle
    import os
    from os import path
    import datetime
    import re



    match = re.search(r'\.parquet$', df_name)
    if match:
        df_name = df_name[:match.start()]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_save_path = path.join(relative_model_dir, f'{df_name}', f'{model_name}_{timestamp}.pkl')
    os.makedirs(path.dirname(model_save_path), exist_ok=True)

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    