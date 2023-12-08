import os

def walk(input_root_dir, output_root_dir, suffix, action):
    for input_folder_path, dirs, files in os.walk(input_root_dir, topdown=False):
        sub_folder_dir = os.path.relpath(input_folder_path, input_root_dir)
        output_folder_path = os.path.join(output_root_dir, sub_folder_dir)
        os.makedirs(output_folder_path, exist_ok=True)

        for full_file_name in files:
            file_name, file_extension = os.path.splitext(full_file_name)
            input_file_path = os.path.join(input_folder_path, full_file_name)
            output_file_path = os.path.join(output_folder_path, f"{file_name}_{suffix}{file_extension}")
            action(input_file_path, output_file_path)