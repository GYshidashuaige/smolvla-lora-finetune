"""Retryable range downloader for public upstream artifacts."""
import concurrent.futures
import os
from pathlib import Path
import time
import urllib.request
import requests

def fetch(url, dest, size):
    for key, value in urllib.request.getproxies_registry().items():
        if key in ('http', 'https'):
            os.environ.setdefault(key.upper() + '_PROXY', value)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return
    parts = dest.parent / (dest.name + '.parts')
    parts.mkdir(exist_ok=True)
    chunk = 8 * 1024 * 1024
    def part(i):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        path = parts / str(i)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(8):
            try:
                sep = '&' if '?' in url else '?'
                with requests.get(url + sep + 't=' + str(time.time()),
                                  headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 20)) as r:
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise RuntimeError('Server did not honor byte range')
                    with path.open('wb') as f:
                        for block in r.iter_content(256 * 1024):
                            f.write(block)
                if path.stat().st_size != end - start + 1:
                    raise RuntimeError('Incomplete range')
                print(f'{dest.name}: part {i + 1}/{(size + chunk - 1) // chunk}', flush=True)
                return path
            except Exception as e:
                print(f'retry part {i}: {type(e).__name__}', flush=True)
                if attempt == 7:
                    raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(part, range((size + chunk - 1) // chunk)))
    import shutil
    temp = dest.with_name(dest.name + '.assembling')
    with temp.open('wb') as output:
        for path in paths:
            with path.open('rb') as source:
                shutil.copyfileobj(source, output)
    assert temp.stat().st_size == size
    temp.replace(dest)

if __name__ == '__main__':
    import sys
    fetch(sys.argv[1], sys.argv[2], int(sys.argv[3]))
