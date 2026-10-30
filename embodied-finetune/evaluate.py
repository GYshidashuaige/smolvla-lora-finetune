"""Paired held-out flow-matching loss, not closed-loop task success."""
import argparse
import json
import os
from pathlib import Path
import urllib.request

for key, value in urllib.request.getproxies_registry().items():
    if key in ('http', 'https'):
        os.environ.setdefault(key.upper() + '_PROXY', value)

import torch
from torch.utils.data import default_collate
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_pre_post_processors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = 'cuda'
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str(root / 'cache/vlm')
    meta = LeRobotDatasetMetadata('lerobot/libero', root=root / 'cache/libero')
    dataset = LeRobotDataset('lerobot/libero', root=root / 'cache/libero', episodes=[40, 41, 42, 43],
                            delta_timestamps=resolve_delta_timestamps(cfg, meta), video_backend='pyav')
    policy = make_policy(cfg, ds_meta=meta).eval()
    processor, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    rows = []
    indices = torch.linspace(0, len(dataset) - 1, args.samples).long().tolist()
    for index in indices:
        batch = processor(default_collate([dataset[index]]))
        values = {}
        for label in ('base', 'adapted'):
            torch.manual_seed(10000 + index)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                if label == 'base':
                    with policy.disable_adapter():
                        loss, _ = policy.forward(dict(batch))
                else:
                    loss, _ = policy.forward(dict(batch))
            values[label] = float(loss)
        rows.append({'dataset_index': index, **values})
        print(rows[-1], flush=True)
    report = {'metric': 'held_out_flow_matching_loss', 'episodes': [40, 41, 42, 43],
              'training_episodes': list(range(40)), 'sample_count': len(rows),
              'base_mean': sum(r['base'] for r in rows) / len(rows),
              'adapted_mean': sum(r['adapted'] for r in rows) / len(rows),
              'checkpoint': str(args.checkpoint), 'samples': rows,
              'limitations': 'Fixed small episode holdout; same stochastic noise per pair. Not action MSE, not closed-loop success rate.'}
    (root / 'results/held-out.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
