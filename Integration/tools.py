from pathlib import Path
from queue import Queue
from typing import Union

from commons.CONSTANTS import Color
from definitions import ROOT_DIR


def __test_valid_path(path):
    return type(path) is str or type(path) is Path


def bfs_search_for_file(file_name: Union[Path, str]):
    if not __test_valid_path(file_name):
        raise Exception(f"Unsupported file type {type(file_name)}")
    if type(file_name) is str:
        file_name = Path(file_name)

    # Define root path
    root_path = Path(ROOT_DIR)
    # Create a queue for BFS
    q = Queue()
    # Add the root path to the queue
    q.put(root_path.parent)

    while not q.empty():
        # Get the next directory to search
        current_dir = q.get()

        # Search for the file in the current directory
        file_path = current_dir / file_name
        if file_path.exists():
            return file_path

        # Add any subdirectories to the queue
        for sub_dir in current_dir.iterdir():
            if sub_dir.is_dir():
                q.put(sub_dir)

    # If the file was not found, return None
    return None


def get_colors():
    return [color.value for color in Color]
