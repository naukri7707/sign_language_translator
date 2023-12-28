import helpers.file_walker as fw

from .split_data import split
from .lmdata_generator import generate_lmdata
from .flow_video_generator import generate_flow_videos

def split_dataset(src_dir: str):
    split(src_dir, train=3, val=1, test=1)

def generate_all_lmdatas(src_dir: str, skip: int = 0):
    fw.walk(
        src_dir,
        [
            (f'{src_dir}_lmdata', lambda name: f"{name}_lmdata.json"),
        ],
        generate_lmdata,
        skip = skip,
        )
    
def generate_all_flow_videos(src_dir: str, skip: int = 0):
    fw.walk(
        src_dir,
        [
            (f'{src_dir}_lmdata', lambda name: f"{name}_lmdata.json"),
            (f'{src_dir}_hand', lambda name: f"{name}_hand.mp4"),
        ],
        generate_flow_videos,
        skip = skip,
        )