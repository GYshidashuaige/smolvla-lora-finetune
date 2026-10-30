"""Low-memory SmolVLA LoRA launcher; uses upstream training and real LIBERO data."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.episodes, a.rank) < 1 or a.episodes > 1693:
        p.error('steps/rank must be positive; episodes must be 1..1693')
    run = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'outputs' / f'lora-r{a.rank}-{run}'
    cmd = [sys.executable, str(ROOT / 'train_entry.py'),
        f'--policy.path={ROOT / "cache/base-model"}', '--policy.device=cuda',
        '--policy.input_features=null', '--policy.load_vlm_weights=false',
        f'--policy.vlm_model_name={ROOT / "cache/vlm"}',
        '--policy.push_to_hub=false', '--policy.use_amp=true',
        '--policy.resize_imgs_with_padding=[256,256]',
        '--policy.optimizer_lr=0.001',
        f'--policy.scheduler_warmup_steps={min(50, max(1, a.steps // 10))}',
        f'--policy.scheduler_decay_steps={max(a.steps, 100)}',
        '--dataset.repo_id=lerobot/libero',
        f'--dataset.root={ROOT / "cache/libero"}',
        '--dataset.episodes=' + json.dumps(list(range(a.episodes))),
        '--dataset.video_backend=pyav',
        '--dataset.image_transforms.enable=' + str(a.augment).lower(),
        '--batch_size=1', '--num_workers=0', f'--steps={a.steps}',
        '--eval_freq=0', '--log_freq=10', f'--save_freq={min(a.steps, 500)}',
        '--wandb.enable=false', '--seed=42',
        f'--peft.r={a.rank}', f'--output_dir={output}']
    env = os.environ.copy()
    proxies = (urllib.request.getproxies_registry() if os.name == 'nt' else urllib.request.getproxies())
    for scheme, proxy in proxies.items():
        if scheme in ('http', 'https'):
            env.setdefault(scheme.upper() + '_PROXY', proxy)
    env['HF_HOME'] = str(ROOT / 'cache' / 'huggingface')
    env['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    print(subprocess.list2cmdline(cmd), flush=True)
    if a.dry_run:
        return
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    (logs / f'{run}-command.json').write_text(json.dumps(cmd, indent=2), encoding='utf-8')
    with (logs / f'{run}-train.log').open('w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            f.write(line)
            f.flush()
        code = proc.wait()
    (logs / f'{run}-status.json').write_text(json.dumps({
        'exit_code': code, 'steps_requested': a.steps, 'episodes': a.episodes,
        'rank': a.rank, 'augmentation': a.augment, 'output': str(output),
        'note': 'Training loss is not a robot task success rate.'}, indent=2), encoding='utf-8')
    sys.exit(code)

if __name__ == '__main__':
    main()
