"""Summarize actual upstream logs; never infer task success from training loss."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
reports = []
for path in sorted((root / 'logs').glob('*-train.log')):
    text = path.read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    rows = []
    for line in text.splitlines():
        step = re.search(r'step:(\S+)', line)
        loss = re.search(r'loss:([0-9.eE+-]+)', line)
        if step and loss:
            rows.append({'step': step.group(1), 'loss': float(loss.group(1))})
    status = path.with_name(path.name.replace('-train.log', '-status.json'))
    reports.append({'log': path.name, 'status': json.loads(status.read_text()) if status.exists() else 'running or interrupted',
                    'training_metrics': rows, 'task_success_rate': None})
out = root / 'results'
out.mkdir(exist_ok=True)
(out / 'runs.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(reports, ensure_ascii=False, indent=2))
