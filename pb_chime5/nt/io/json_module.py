
import io
import numpy as np
import json
import datetime
from pathlib import Path


# http://stackoverflow.com/a/27050186
class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif str(type(obj)) == "<class 'chainer.variable.Variable'>":
            return obj.num.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif str(type(obj)) == "<class 'bson.objectid.ObjectId'>":
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d_%H-%M-%S')
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return super().default(obj)


class SummaryEncoder(Encoder):
    """
    Often, you may want a very short summary, just containing some shapes of
    numpy arrays in a Jupyter Notebook.

    Example usage:
    >>> import numpy as np
    >>> example = dict(a=np.random.uniform(size=(3, 4)))
    >>> print(json.dumps(example, cls=SummaryEncoder, indent=2))
    {
      "a": "ndarray: shape (3, 4), dtype float64"
    }

    alternative:
    >>> np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')
    >>> example
    {'a': array(shape=(3, 4), dtype=float64)}
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return 'ndarray: shape {}, dtype {}'.format(obj.shape, obj.dtype)
        else:
            return super().default(obj)


def dumps_json(
        obj, *, indent=2, sort_keys=True, **kwargs):
    fd = io.StringIO()
    dump_json(
        obj,
        path=fd,
        indent=indent,
        create_path=False,
        sort_keys=sort_keys,
        **kwargs,
    )
    return fd.getvalue()


def dump_json(
        obj, path, *, indent=2, create_path=True, sort_keys=True, **kwargs):
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
                  sort_keys=sort_keys, **kwargs)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()

        if create_path:
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            json.dump(obj, f, cls=Encoder, indent=indent,
                      sort_keys=sort_keys, **kwargs)
    else:
        raise TypeError(path)


def load_json(path, **kwargs):
    """ Loads a JSON file and returns it as a dict.

    :param path: String or ``pathlib.Path`` object.
    :param kwargs: See ``json.dump()``.
    :return: Content of the JSON file.
    """
    assert isinstance(path, (str, Path)), path
    path = Path(path).expanduser()

    with path.open() as fid:
        return json.load(fid, **kwargs)


def loads_json(fid, **kwargs):
    """ Loads a JSON file and returns it as a dict.

    :param path: String or another object that is accepted by json.loads
    :param kwargs: See ``json.dump()``.
    :return: Content of the JSON file.
    """
    assert isinstance(fid, str), fid

    return json.loads(fid, **kwargs)
