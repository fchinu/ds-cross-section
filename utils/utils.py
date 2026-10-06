"""
Utility functions for the analysis.
"""
from pathlib import Path
import shutil
import sys

def create_dir(directory):
    """Create a directory if it doesn't exist, or overwrite it if it does."""
    directory = Path(directory)
    if not directory.exists():
        print(f"\033[32m{directory} does not exist, it will be created\033[0m")
    else:
        print(f"\033[33m{directory} already exists, it will be overwritten\033[0m")
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    return directory

def logger(message, level='INFO'):
    """
    Function to log messages with different levels.
    Args:
        message (str): The message to log.
        level (str): The level of the message ('INFO', 'WARNING', 'ERROR').
    """
    message = f"[{level}] {message}"
    if level == 'INFO':
        print(f"\033[32m{message}\033[0m")
    elif level == 'WARNING':
        print(f"\033[33m{message}\033[0m")
    elif level == 'ERROR':
        print(f"\033[31m{message}\033[0m")
        sys.exit(1)
    elif level == 'COMMAND':
        print(f"\033[35m{message}\033[0m")
    elif level == 'DEBUG':
        print(f"\033[34m{message}\033[0m")
    elif level == 'PAUSE':
        input(f"\033[36m{message}\n{level}: Press Enter to continue.\033[0m")
    else:
        print(f"\033[37m{message}\033[0m")  # Default to white for unknown levels

def enforce_list(var):
    """
    Ensure that the input variable is a list.

    Parameters:
    - var (any): The input variable.

    Returns:
    - list: The input variable as a list.
    """
    if not isinstance(var, list):
        var = [var]
    return var


def get_root_files_in_directory(directory):
    """
    Get all ROOT files in the specified directory.

    Args:
    - directory (str): The directory to search for ROOT files.

    Returns:
    - root_files (list): A list of ROOT file paths.
    """
    path = Path(directory)
    root_files = [str(f) for f in sorted(path.glob('*.root'))]
    return root_files
