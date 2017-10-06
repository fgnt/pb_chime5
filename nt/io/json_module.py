
import io
import numpy as np
import json
import bson
import datetime
from pathlib import Path


# http://stackoverflow.com/a/27050186
class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return self._handle_np_integer(obj)
        elif isinstance(obj, np.floating):
            return self._handle_np_floating(obj)
        elif str(type(obj)) == "<class 'chainer.variable.Variable'>":
            return self._handle_chainer_variable(obj)
        elif isinstance(obj, np.ndarray):
            return self._handle_np_ndarray(obj)
        elif isinstance(obj, bson.objectid.ObjectId):
            return self._handle_bson_object_id(obj)
        elif isinstance(obj, datetime.datetime):
            return self._handle_datetime_datetime(obj)
        else:
            return super().default(obj)

    @staticmethod
    def _handle_np_integer(obj):
        return int(obj)

    @staticmethod
    def _handle_np_floating(obj):
        return float(obj)

    @staticmethod
    def _handle_chainer_variable(obj):
        return obj.num.tolist()

    @staticmethod
    def _handle_np_ndarray(obj):
        return obj.tolist()

    @staticmethod
    def _handle_bson_object_id(obj):
        return str(obj)

    @staticmethod
    def _handle_datetime_datetime(obj):
        return obj.strftime('%Y-%m-%d_%H-%M-%S')


class SummaryEncoder(Encoder):
    """
    Often, you may want a very short summary, just containing some shapes of
    numpy arrays in a Jupyter Notebook.

    Example usage:
    >>> import numpy as np
    >>> example = dict(a=np.random.uniform(size=(3, 4))
    >>> print(json.dumps(example, cls=SummaryEncoder, indent=2))
    """
    @staticmethod
    def _handle_np_ndarray(obj):
        return 'ndarray: shape {}, dtype {}'.format(obj.shape, obj.dtype)


def dump_json(obj, path, *, indent=2, **kwargs):
    """
    Numpy types will be converted to the equivalent Python type for dumping the
    object.

    :param obj: Arbitrary object that is JSON serializable,
        where Numpy is allowed.
    :param path: String or ``pathlib.Path`` object.
    :param indent: See ``json.dump()``.
    :param kwargs: See ``json.dump()``.

    """
    if isinstance(path, io.IOBase):
        json.dump(obj, path, cls=Encoder, indent=indent,
                  sort_keys=True, **kwargs)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()

        with path.open('w') as f:
            json.dump(obj, f, cls=Encoder, indent=indent,
                      sort_keys=True, **kwargs)
    else:
        raise TypeError(path)


def load_json(path, **kwargs):
    """ Loads a JSON file and returns it as a dict.

    :param path: String or ``pathlib.Path`` object.
    :param kwargs: See ``json.dump()``.
    :return: Content of the JSON file.
    """
    assert isinstance(path, (str, Path))
    path = Path(path).expanduser()

    with path.open() as fid:
        return json.load(fid, **kwargs)
