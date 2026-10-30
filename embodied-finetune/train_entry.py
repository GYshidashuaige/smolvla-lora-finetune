"""Upstream training with a Windows-safe last-checkpoint directory."""
import os
import shutil
from pathlib import Path

from lerobot.utils import train_utils

if os.name == 'nt':
    def update_last_checkpoint(checkpoint_dir):
        source = Path(checkpoint_dir).resolve()
        expected_root = (Path(__file__).resolve().parent / 'outputs').resolve()
        if not source.is_relative_to(expected_root):
            raise ValueError('Checkpoint must stay inside this project outputs directory')
        target = source.parent / 'last'
        # No administrator symlink privilege required. Each checkpoint retains its numbered directory.
        if target.is_symlink():
            target.unlink()
        shutil.copytree(source, target, dirs_exist_ok=True)
    train_utils.update_last_checkpoint = update_last_checkpoint

from lerobot.scripts.lerobot_train import main

if __name__ == '__main__':
    main()
