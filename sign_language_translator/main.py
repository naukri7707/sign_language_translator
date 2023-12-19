import file_walker as fw
import flow_image_generactor as fig

fw.walk(
    '~data/2_360p',
    [
        ('~data/3_lmdata', lambda name: f"{name}_lmdata.json"),
        ('~data/4_frames', lambda name: f"{name}_frames"),
    ],
    fig.videos_to_frames,
    skip=2120,
)