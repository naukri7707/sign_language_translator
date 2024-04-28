import time
from pathlib import Path
from typing import Callable


def traverse_files(
    directory: str,
    *,
    on_traversed: Callable[[Path], None],
    filter: Callable[[Path], bool] = lambda _: True,
) -> int:
    directory_path = Path(directory)
    total_file_count = 0
    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and filter(file_path):
            total_file_count += 1

    total_file_count_digits = len(str(total_file_count))

    file_count = 0
    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and filter(file_path):
            start_time = time.time()
            on_traversed(file_path)
            end_time = time.time()
            elapsed_time = end_time - start_time

            file_count += 1
            print(
                f"[{file_count:0{total_file_count_digits}d}/{total_file_count}]: Processing completed '{file_path}' in {elapsed_time:.2f} seconds."
            )

    return file_count
