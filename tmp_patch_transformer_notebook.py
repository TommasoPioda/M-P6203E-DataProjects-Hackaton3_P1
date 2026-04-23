from pathlib import Path
import json

path = Path('Models/graph_based/notebooks/Transformer.ipynb')
data = json.loads(path.read_text(encoding='utf-8'))
changed = False
for cell in data['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'save_model_artifact = model_saver.save_model_artifact' in src:
        cell['source'] = [
            'import importlib\n',
            'import utils.textual_utils.registry.legacy_model_saver as legacy_model_saver\n',
            'importlib.reload(legacy_model_saver)\n',
            'save_model_artifact = legacy_model_saver.save_model_artifact\n',
            '\n',
            'df_name = "exploded_splits"\n',
            '\n',
            'model_dir, summary_path = save_model_artifact(\n',
        ]
        changed = True
        break
if not changed:
    raise SystemExit('Target cell not found')
path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
print('notebook patched')
