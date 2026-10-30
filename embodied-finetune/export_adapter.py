"""Export a portable adapter without local base-model paths or optimizer states."""
import argparse
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
target = root / 'artifacts/adapter'
target.mkdir(parents=True, exist_ok=True)
for path in args.checkpoint.iterdir():
    if path.name in ('train_config.json', 'README.md') or not path.is_file():
        continue
    shutil.copy2(path, target / path.name)
for path in target.glob('*.json'):
    data = json.loads(path.read_text(encoding='utf-8'))
    if path.name == 'adapter_config.json':
        data['base_model_name_or_path'] = 'lerobot/smolvla_base'
    if path.name == 'config.json':
        data['vlm_model_name'] = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
(target / 'README.md').write_text('''---
license: apache-2.0
base_model: lerobot/smolvla_base
library_name: peft
tags: [robotics, smolvla, lora]
---
# SmolVLA LIBERO small-data LoRA experiment

Small-data LoRA adaptation on LIBERO. Exact steps, episode split, rank, batch size,
and seed are recorded in the accompanying project logs and results.
This adapter is an experimental artifact. No closed-loop task success is claimed.
See the project logs and results for measured evidence and limitations.
Base model and dataset retain their original licenses and terms.
''', encoding='utf-8')
print(target)
