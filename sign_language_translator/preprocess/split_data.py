

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

    start = 0
    for tag, ratio in tag_ratio_pair.items():
        subsets[tag] = {
            'file_range': range(start, start + ratio - 1),
            'root_path': target_root_path / tag 
        }
        start += ratio

    for dir in sub_dirs:
        items = list(dir.iterdir())
        shuffle(items)
        for i, item_path in enumerate(items):
            relative_path = item_path.relative_to(root_path)
            for key, value in subsets.items():
                if i in value['file_range']:
                    target_path = value['root_path'] / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(item_path, target_path)
                    break


split('~data/1~10', train=3, val=1, test=1)