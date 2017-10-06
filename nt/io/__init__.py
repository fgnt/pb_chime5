"""
This module deals with all sorts of input and output.

There is special focus on audio, but there are also some convenience imports
i.e. for load_json() and similar functions.

The file path is called `path` just as it has been done in ``audioread``.
The path should either be a ``pathlib.Path`` object or a string.
"""
import pickle
from pathlib import Path

from nt.io import audioread
from nt.io import hdf5
from nt.io import play
from nt.io import json_module as json
from nt.io.json_module import load_json, dump_json
from nt.io.file_handling import mkdir_p

__all__ = [
    'load_json', 'dump_json',
    'load_hdf5', 'dump_hdf5', 'update_hdf5',
    'load_pickle', 'dump_pickle', 'mkdir_p'
]


def load_hdf5(path, internal_path='/'):
    return hdf5.load_hdf5(
        str(Path(path).expanduser()),
        str(internal_path)
    )


def dump_hdf5(data, path):
    assert isinstance(path, (str, Path))

    hdf5.dump_hdf5(
        data,
        str(Path(path).expanduser())
    )


def update_hdf5(data, path, prefix='/'):
    assert isinstance(path, (str, Path, hdf5.h5py.File))

    if isinstance(path, hdf5.h5py.File):
        hdf5.update_hdf5(
            data,
            path,
            path=prefix
        )
    else:
        hdf5.update_hdf5(
            data,
            str(Path(path).expanduser()),
            path=prefix
        )


def load_pickle(path):
    assert isinstance(path, (str, Path))
    path = Path(path).expanduser()
    with path.open('rb') as f:
        return pickle.load(f)


def dump_pickle(data, path):
    assert isinstance(path, (str, Path))
    path = Path(path).expanduser()
    with path.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
