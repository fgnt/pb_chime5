import numpy as np


def recursive_load_decorator(default_list_to='list'):
    """
    >>> from pb_chime5.io import load_audio
    >>> np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')
    >>> load_audio = recursive_load_decorator(default_list_to='array')(load_audio)
    >>> prefix = '/net/fastdb/chime3/audio/16kHz/isolated/dt05_str_simu'
    >>> data = {
    ...     'a': [
    ...         f'{prefix}/F01_22HC010Q_STR.CH3.wav',
    ...         f'{prefix}/F01_22HC010Q_STR.CH2.wav',
    ...      ],
    ...      'b': f'{prefix}/F01_22HC010Q_STR.CH1.wav',
    ... }
    >>> load_audio(data)
    {'a': array(shape=(2, 160893), dtype=float64), 'b': array(shape=(160893,), dtype=float64)}

    """
    assert isinstance(default_list_to, str), default_list_to

    def decorator(func):
        def wrapper(path, *args, list_to=default_list_to, **kwargs):
            def self_call(nested_path):
                return wrapper(
                    nested_path,
                    *args,
                    list_to=list_to,
                    **kwargs,
                )

            if isinstance(path, (list, tuple)):
                if list_to == 'dict':
                    return {f: self_call(f) for f in path}
                elif list_to == 'array':
                    return np.array([self_call(f) for f in path])
                elif list_to == 'list':
                    return [self_call(f) for f in path]
                else:
                    raise ValueError(list_to)
            elif isinstance(path, (dict,)):
                return path.__class__({
                    k: self_call(v)
                    for k, v in path.items()
                })
            else:
                return func(path, *args, **kwargs)
        return wrapper
    return decorator
