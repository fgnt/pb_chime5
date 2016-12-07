from contextlib import contextmanager
import os
import pathlib


def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists

    :param path: path to create
    :return: None
    """
    if isinstance(path, pathlib.Path):
        path = str(path)

    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except FileNotFoundError:
        if path == '':
            pass


@contextmanager
def change_directory(new_path):
    """ Context manager for changing the current working directory.

    Will return to original directory even on error.

    http://stackoverflow.com/a/24176022/911441

    Args:
        new_path: Directory to change to.

    Returns:

    """
    saved_path = os.getcwd()
    os.chdir(os.path.expanduser(new_path))
    try:
        yield
    finally:
        os.chdir(saved_path)
