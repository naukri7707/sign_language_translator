import helpers.file_walker as fw

from .lmdata_generator import generate_lmdata
from .flow_video_generator import generate_flow_videos

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
            (f'{src_dir}_flow', lambda name: f"{name}_flow"),
        ],
        generate_flow_videos,
        skip = skip,
        )