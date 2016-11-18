"""
This module deals with all sorts of input and output.

There is special focus on audio, but there are also some convenience imports
i.e. for load_json() and similar functions.

The file path is called `path` just as it has been done in audioread.
"""
from nt.io import audioread
from nt.io import play
import json
from pathlib import Path
import bson
import numpy as np
import datetime
from nt.utils import hdf5_utils


__all__ = [
    'load_json', 'dump_json',
    'load_hdf5', 'dump_hdf5'
]


# http://stackoverflow.com/a/27050186
class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bson.objectid.ObjectId):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d_%H-%M-%S')
        else:
            return super().default(obj)


def load_json(path, internal_path='/'):
    assert isinstance(path, (str, Path))
    assert isinstance(internal_path, (str, Path)) or internal_path is None

    with open(path) as fid:
        content = json.load(fid)

    internal_path = str(internal_path)
    for key in internal_path.split('/'):
        content = content[key]

    return content


def dump_json(data, path):
    assert isinstance(path, (str, Path))

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=_Encoder)


def load_hdf5(path, internal_path='/'):
    return hdf5_utils.hdf5_load(
        str(path),
        str(internal_path)
    )


def dump_hdf5(data, path):
    assert isinstance(path, (str, Path))

    hdf5_utils.hdf5_dump(
        data,
        str(path)
    )
