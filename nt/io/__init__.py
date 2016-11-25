"""
This module deals with all sorts of input and output.

There is special focus on audio, but there are also some convenience imports
i.e. for load_json() and similar functions.

The file path is called `path` just as it has been done in ``audioread``.
The path should either be a ``pathlib.Path`` object or a string.
"""
from pathlib import Path

from nt.io import audioread
from nt.io import hdf5
from nt.io import play
from nt.io.json import load_json, dump_json

__all__ = [
    'load_json', 'dump_json',
    'load_hdf5', 'dump_hdf5', 'update_hdf5'
]


def load_hdf5(path, internal_path='/'):
    return hdf5.load_hdf5(
        str(path),
        str(internal_path)
    )


def dump_hdf5(data, path):
    assert isinstance(path, (str, Path))

    hdf5.dump_hdf5(
        data,
        str(path)
    )


def update_hdf5(data, path):
    assert isinstance(path, (str, Path))

    hdf5.update_hdf5(
        data,
        str(path)
    )
