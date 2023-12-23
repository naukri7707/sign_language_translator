import os

walk_root_dir="~data/1_sorted"

def rename_folder(folder_path, name, new_name):
    new_folder_path = folder_path.replace(name, new_name)
    if os.path.exists(folder_path):
        return

    if name in folder_path and not new_name in folder_path:
        os.rename(folder_path, folder_path.replace(name, new_name))

def rename_file(file_path, name, new_name):
    new_file_path = file_path.replace(name, new_name)
    if os.path.exists(new_file_path):
        return

    if name in file_path and not new_name in file_path:
        os.rename(file_path, new_file_path)


for current_folder_path, sub_dirs, files in os.walk(walk_root_dir, topdown=False):
    rename_folder(current_folder_path, '規定', '規定(A)')
    rename_folder(current_folder_path, '新鮮', '新鮮(A)')
    rename_folder(current_folder_path, '圖書館', '圖書館(A)')
    rename_folder(current_folder_path, '端午節', '端午節(A)')
    rename_folder(current_folder_path, '獨角仙', '獨角仙(A)')
    rename_folder(current_folder_path, '螃蟹', '螃蟹(A)')
    rename_file(current_folder_path, '駱駝', '駱駝(A)')

for current_folder_path, sub_dirs, files in os.walk(walk_root_dir, topdown=False):
    for file_path in files:
        rename_file(f"{current_folder_path}/{file_path}", '規定', '規定(A)')
        rename_file(f"{current_folder_path}/{file_path}", '新鮮', '新鮮(A)')
        rename_file(f"{current_folder_path}/{file_path}", '圖書館', '圖書館(A)')
        rename_file(f"{current_folder_path}/{file_path}", '端午節', '端午節(A)')
        rename_file(f"{current_folder_path}/{file_path}", '獨角仙', '獨角仙(A)')
        rename_file(f"{current_folder_path}/{file_path}", '螃蟹', '螃蟹(A)')
        rename_file(f"{current_folder_path}/{file_path}", '駱駝', '駱駝(A)')

for current_folder_path, sub_dirs, files in os.walk(walk_root_dir, topdown=False):
    if not os.listdir(current_folder_path):
        os.rmdir(current_folder_path)