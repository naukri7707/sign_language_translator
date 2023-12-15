import os
import time

LOG_FILE_PATH = "logs/file_walker.log"

def __print_and_log(message):
    print(message)
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(message + "\n")

def __count_paths(input_root_dir) -> (int, int):
    sub_dir_count = 0
    file_count = 0
    for input_folder_path, dirs, files in os.walk(input_root_dir, topdown=False):
        sub_dir_count += 1
        for full_file_name in files:
            file_count += 1
    return (sub_dir_count, file_count)

def walk(input_root_dir, output_root_dir, tag, action, output_extension = None, skip = 0):
    # create 'logs' folder if not exist
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    sub_dir_count, file_count = __count_paths(input_root_dir)
    current_sub_dir_count = 0
    current_file_count = 0

    sub_dir_rjust_len = len(str(sub_dir_count))
    file_rjust_len = len(str(file_count))

    for current_folder_path, sub_dirs, files in os.walk(input_root_dir, topdown=False):
        current_sub_dir_count += 1
        __print_and_log(f"Start process dir: {current_folder_path} ({str(current_sub_dir_count).rjust(sub_dir_rjust_len)}/{sub_dir_count})")

        sub_folder_dir = os.path.relpath(current_folder_path, input_root_dir)
        output_folder_path = os.path.join(output_root_dir, sub_folder_dir)
        os.makedirs(output_folder_path, exist_ok=True)
        for full_file_name in files:

            current_file_count += 1
            start_time = time.time()
            
            file_name, file_extension = os.path.splitext(full_file_name)
            input_file_path = os.path.join(current_folder_path, full_file_name)
            output_file_name = f"{file_name}_{tag}{output_extension if output_extension else file_extension}"
            output_file_path = os.path.join(output_folder_path, output_file_name)

            if current_file_count <= skip:
                __print_and_log(f'[{str(current_file_count).rjust(file_rjust_len)}/{str(file_count).rjust(file_rjust_len)}] Skip process "{output_file_name}".')
                continue
            else:
                action(input_file_path, output_file_path)

            end_time = time.time()
            execution_time = end_time - start_time
            __print_and_log(f'[{str(current_file_count).rjust(file_rjust_len)}/{str(file_count).rjust(file_rjust_len)}] Completed process "{output_file_name}" in {round(execution_time, 2)}s.')
    __print_and_log(f"Done!")