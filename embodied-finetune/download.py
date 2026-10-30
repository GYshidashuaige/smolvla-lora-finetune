"""Download original weights and real data, honoring Windows system proxy."""
import os
from pathlib import Path
import urllib.request
import time
import requests

proxies = (urllib.request.getproxies_registry() if os.name == 'nt' else urllib.request.getproxies())
for scheme, proxy in proxies.items():
    if scheme in ('http', 'https'):
        os.environ.setdefault(scheme.upper() + '_PROXY', proxy)
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent / 'cache'
weights = root / 'base-model/model.safetensors'
if not weights.exists():
    weights.parent.mkdir(parents=True, exist_ok=True)
    # Cache-busting avoids expired signed CDN redirects from intermediary caches.
    url = 'https://huggingface.co/lerobot/smolvla_base/resolve/main/model.safetensors?download=true&t=' + str(time.time())
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        temp = weights.with_suffix('.partial')
        with temp.open('wb') as output:
            for block in response.iter_content(1024 * 1024):
                output.write(block)
        expected = response.headers.get('content-length')
        if expected and temp.stat().st_size != int(expected):
            raise RuntimeError('Incomplete model download')
        temp.replace(weights)
snapshot_download('lerobot/smolvla_base', local_dir=root / 'base-model', ignore_patterns=['model.safetensors'])
snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
                  local_dir=root / 'vlm', allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
for attempt in range(5):
    try:
        snapshot_download('lerobot/libero', repo_type='dataset', local_dir=root / 'libero',
                          allow_patterns=['meta/*', 'meta/**/*', 'data/*/*', 'videos/*/chunk-000/file-000.mp4'],
                          max_workers=4)
        break
    except Exception:
        if attempt == 4:
            raise
        print('Retrying dataset download; completed files are reused.', flush=True)
