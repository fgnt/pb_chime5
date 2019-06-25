import itertools


def merge(*dicts):
    """
    Merge dicts with orthogonal keys. It is only allowed, that all dicts
    have the same key, when the corresponding values are equal.
    Keep the type from the first dict.

    >>> d1 = {'a': 1}
    >>> d2 = {'b': 2}
    >>> d3 = {'a': 3}
    >>> merge(d1, d2)
    {'a': 1, 'b': 2}
    >>> merge(d1, d1)
    {'a': 1}
    >>> merge(d1, d3)
    Traceback (most recent call last):
    ...
    AssertionError: Item for key 'a' is not unique.
    In the 0. dict is the value 1, while in the merged 3.
    Intersection keys: {'a'}
    >>> dict(**d1, **d2)
    {'a': 1, 'b': 2}
    >>> dict(**d1, **d3)
    Traceback (most recent call last):
    ...
    TypeError: type object got multiple values for keyword argument 'a'
    """
    intersection = set.intersection(*[
        set(d.keys())
        for d in dicts
    ])

    dict_merged = dicts[0].__class__(
        # collections.ChainMap(*reversed(dicts))  # does not keep order
        itertools.chain(*[d.items() for d in dicts])
    )
    if len(intersection) != 0:
        for k in intersection:
            for i, d in enumerate(dicts):
                if dict_merged[k] != d[k]:
                    raise AssertionError(
                        f"Item for key {k!r} is not unique.\n"
                        f"In the {i}. dict is the value {d[k]!r}, "
                        f"while in the merged {dict_merged[k]!r}.\n"
                        f"Intersection keys: {intersection}"
                    )
    # assert len(intersection) == 0, intersection
    return dict_merged
