import ffmpeg
import file_enumerate as fe

def get_resolution(input_file):
    probe = ffmpeg.probe(input_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def process_video(input_file, output_file):
    width, height = get_resolution(input_file)
    
    if width == 1920 and height == 1080:
        ffmpeg.input(input_file).output(output_file, vf='scale=640:360', b='4096k').run()
    elif width == 640 and height == 480:
        ffmpeg.input(input_file).output(output_file, vf='crop=640:360:0:60', b='4096k').run()
    else:
        print("Unsupported resolution")

fe.walk(
    '../~data/renamed',
    '../~data/360p', 
    '360p',
    process_video
    )