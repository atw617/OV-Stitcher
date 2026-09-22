"""Evaluate OV-Stitcher on all configured datasets, one at a time."""

import subprocess
import sys
from pathlib import Path


CONFIGS = (
    'cfg_ade20k.py',
    'cfg_city_scapes.py',
    'cfg_coco_object.py',
    'cfg_coco_stuff164k.py',
    'cfg_context59.py',
    'cfg_context60.py',
    'cfg_voc20.py',
    'cfg_voc21.py',
)


def main():
    root = Path(__file__).resolve().parent
    for config in CONFIGS:
        config_path = root / 'configs' / config
        print(f'Running {config_path}', flush=True)
        subprocess.run(
            [sys.executable, str(root / 'eval.py'), '--config', str(config_path)],
            cwd=root,
            check=True,
        )


if __name__ == '__main__':
    main()
