import os
import glob

source_directory = '~data/1_sorted'
dst_directory = '~data/1_sorted'

# 將篩選出來的資料夾移動到對應的目標資料夾中
def move_folder(tag,folders):
    root_folder = f"{dst_directory}/{tag}"

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for src_path in folders:
        path = os.path.basename(src_path)

        parts = path.split('_')
        label = parts[0]

        true_path = os.path.join(root_folder, label)
        if not os.path.exists(true_path):
            os.makedirs(true_path)
        
        true_path = os.path.join(true_path, path)

        os.rename(src_path, true_path)

# 使用 glob 篩選含有特定字串的資料夾
top_folders = glob.glob(os.path.join(source_directory, '**/*top*'))
front_folders = glob.glob(os.path.join(source_directory, '**/*front*'))

move_folder('top', top_folders)
move_folder('front', front_folders)

# 刪除空資料夾
subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
for subdir in subdirectories:
    path = f"{source_directory}/{subdir}"
    if not os.listdir(path):
        os.rmdir(path)

