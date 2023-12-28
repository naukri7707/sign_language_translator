

import os
import shutil
from pathlib import Path
from random import shuffle

from matplotlib.dates import relativedelta


def split(dataset_path: str, **tag_ratio_pair):
   
    root_path = Path(dataset_path)

    sub_dirs = [subdir for subdir in root_path.iterdir() if subdir.is_dir()]

    target_root_path = Path(dataset_path + '_split')

    subsets = {}

    ratio_sum = 0
    for tag, ratio in tag_ratio_pair.items():
        subsets[tag] = {
            'file_range': range(ratio_sum, ratio_sum + ratio),
            'root_path': target_root_path / tag 
        }
        ratio_sum += ratio

    for dir in sub_dirs:
        items = list(dir.iterdir())
        shuffle(items)
        for i, item_path in enumerate(items):
            bkt_idx = i % ratio_sum
            relative_path = item_path.relative_to(root_path)
            for key, value in subsets.items():
                if bkt_idx in value['file_range']:
                    target_path = value['root_path'] / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(item_path, target_path)
                    break