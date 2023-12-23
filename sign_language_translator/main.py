import file_walker as fw
import flow_image_generactor as fig

fw.walk(
    '~data/2_360p',
    [
        ('~data/3_lmdata', lambda name: f"{name}_lmdata.json"),
        ('~data/4_step3', lambda name: f"{name}_step3"),
    ],
    fig.videos_to_frames,
    skip=0,
)