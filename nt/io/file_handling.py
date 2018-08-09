from contextlib import contextmanager
import os
from pathlib import Path


def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists

    :param path: path to create
    :return: None
    """
    if isinstance(path, Path):
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


def symlink(source, link_name):
    """
    Create a link to source at link_name.
    Similar to "ln -s <source> <link_name>"

    Special properties:
     - Try os.symlink
     - except link_name already point to source: pass
     - except link_name exists and point somewhere else: ImproveExceptionMsg
     - except link_name exists and is not a symlink: reraise
     - except link_name.parent does not exists: ImproveExceptionMsg

    """
    try:
        os.symlink(source, link_name)
    except FileExistsError:
        link_name = Path(link_name)

        # link_name.exists() is False when the link does not exist
        if link_name.is_symlink():
            link = os.readlink(link_name)
            if link == str(source):
                pass
            else:
                raise FileExistsError(
                    'File exist.\n'
                    f'Try:       {source} -> {link_name}\n'
                    f'Currently: {link} -> {link_name}'
                )
        elif link_name.exists():
            raise
        else:
            assert not link_name.parent.exists(), 'Should not happen.'
            raise FileNotFoundError(
                f'The parent directory of the dst {link_name} does not exist'
                f'{link_name}'
            )
