import os


def count_files_on(data_dir):
    """Count how many files are in the data dir, useful to not bound samples to a specific amount"""
    _path, _dirs, files = next(os.walk(data_dir))
    return len(files)
