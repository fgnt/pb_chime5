import re
import warnings
import numpy as np
import collections
import numbers
from numpy.core.einsumfunc import _parse_einsum_input
from dataclasses import dataclass


def segment_axis_v2(x, length: int, shift: int, axis: int=-1,
                    end='pad', pad_mode='constant', pad_value=0):
    """ !!! WIP !!!

    ToDo: Discuss: Outsource conv_pad?

    Generate a new array that chops the given array along the given axis
    into overlapping frames.

    Note: if end='pad' the return is maybe a copy

    :param x: The array to segment
    :param length: The length of each frame
    :param shift: The number of array elements by which the frames should shift
        Negative values are also allowed.
    :param axis: The axis to operate on
    :param end:
        'pad' -> pad,
            pad the last block with zeros if necessary
        None -> assert,
            assume the length match, ensures a no copy
        'cut' -> cut,
            remove the last block if there are not enough values
        'conv_pad'
            special padding for convolution, assumes shift == 1, see example
            below

    :param pad_mode: see numpy.pad
    :param pad_value: The value to pad
    :return:

    >>> # import cupy as np
    >>> segment_axis_v2(np.arange(10), 4, 2)  # simple example
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    >>> segment_axis_v2(np.arange(10), 4, -2)  # negative shift
    array([[6, 7, 8, 9],
           [4, 5, 6, 7],
           [2, 3, 4, 5],
           [0, 1, 2, 3]])
    >>> segment_axis_v2(np.arange(5).reshape(5), 4, 1, axis=0)
    array([[0, 1, 2, 3],
           [1, 2, 3, 4]])
    >>> segment_axis_v2(np.arange(5).reshape(5), 4, 2, axis=0, end='cut')
    array([[0, 1, 2, 3]])
    >>> segment_axis_v2(np.arange(5).reshape(5), 4, 2, axis=0, end='pad')
    array([[0, 1, 2, 3],
           [2, 3, 4, 0]])
    >>> segment_axis_v2(np.arange(5).reshape(5), 4, 1, axis=0, end='conv_pad')
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 2],
           [0, 1, 2, 3],
           [1, 2, 3, 4],
           [2, 3, 4, 0],
           [3, 4, 0, 0],
           [4, 0, 0, 0]])
    >>> segment_axis_v2(np.arange(6).reshape(6), 4, 2, axis=0, end='pad')
    array([[0, 1, 2, 3],
           [2, 3, 4, 5]])
    >>> segment_axis_v2(np.arange(10).reshape(2, 5), 4, 1, axis=-1)
    array([[[0, 1, 2, 3],
            [1, 2, 3, 4]],
    <BLANKLINE>
           [[5, 6, 7, 8],
            [6, 7, 8, 9]]])
    >>> segment_axis_v2(np.arange(10).reshape(5, 2).T, 4, 1, axis=1)
    array([[[0, 2, 4, 6],
            [2, 4, 6, 8]],
    <BLANKLINE>
           [[1, 3, 5, 7],
            [3, 5, 7, 9]]])
    >>> segment_axis_v2(np.asfortranarray(np.arange(10).reshape(2, 5)),
    ...                 4, 1, axis=1)
    array([[[0, 1, 2, 3],
            [1, 2, 3, 4]],
    <BLANKLINE>
           [[5, 6, 7, 8],
            [6, 7, 8, 9]]])
    >>> segment_axis_v2(np.arange(8).reshape(2, 2, 2).transpose(1, 2, 0),
    ...                 2, 1, axis=0, end='cut')
    array([[[[0, 4],
             [1, 5]],
    <BLANKLINE>
            [[2, 6],
             [3, 7]]]])
    >>> a = np.arange(7).reshape(7)
    >>> b = segment_axis_v2(a, 4, -2, axis=0, end='cut')
    >>> a += 1  # a and b point to the same memory
    >>> b
    array([[3, 4, 5, 6],
           [1, 2, 3, 4]])

    >>> segment_axis_v2(np.arange(7), 8, 1, axis=0, end='pad').shape
    (1, 8)
    >>> segment_axis_v2(np.arange(8), 8, 1, axis=0, end='pad').shape
    (1, 8)
    >>> segment_axis_v2(np.arange(9), 8, 1, axis=0, end='pad').shape
    (2, 8)
    >>> segment_axis_v2(np.arange(7), 8, 2, axis=0, end='cut').shape
    (0, 8)
    >>> segment_axis_v2(np.arange(8), 8, 2, axis=0, end='cut').shape
    (1, 8)
    >>> segment_axis_v2(np.arange(9), 8, 2, axis=0, end='cut').shape
    (1, 8)

    >>> x = np.arange(1, 10)
    >>> filter_ = np.array([1, 2, 3])
    >>> np.convolve(x, filter_)
    array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
    >>> x_ = segment_axis_v2(x, len(filter_), 1, end='conv_pad')
    >>> x_
    array([[0, 0, 1],
           [0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9],
           [8, 9, 0],
           [9, 0, 0]])
    >>> x_ @ filter_[::-1]  # Equal to convolution
    array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
    """
    
    if x.__class__.__module__ == 'cupy.core.core':
        import cupy
        xp = cupy
    else:
        xp = np

    axis = axis % x.ndim

    # Implement negative shift with a positive shift and a flip
    # stride_tricks does not work correct with negative stride
    if shift > 0:
        do_flip = False
    elif shift < 0:
        do_flip = True
        shift = abs(shift)
    else:
        raise ValueError(shift)

    if pad_mode == 'constant':
        pad_kwargs = {'constant_values': pad_value}
    else:
        pad_kwargs = {}

    # Pad
    if end == 'pad':
        if x.shape[axis] < length:
            npad = np.zeros([x.ndim, 2], dtype=np.int)
            npad[axis, 1] = length - x.shape[axis]
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
        elif shift != 1 and (x.shape[axis] + shift - length) % shift != 0:
            npad = np.zeros([x.ndim, 2], dtype=np.int)
            npad[axis, 1] = shift - ((x.shape[axis] + shift - length) % shift)
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)

    elif end == 'conv_pad':
        assert shift == 1, shift
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, :] = length - shift
        x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
    elif end is None:
        assert (x.shape[axis] + shift - length) % shift == 0, \
            '{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})' \
            ''.format((x.shape[axis] + shift - length) % shift,
                      x.shape[axis], shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    # Calculate desired shape and strides
    shape = list(x.shape)
    # assert shape[axis] >= length, shape
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
    shape.insert(axis + 1, length)

    strides = list(x.strides)
    strides.insert(axis, shift * strides[axis])

    # Alternative to np.ndarray.__new__
    # I am not sure if np.lib.stride_tricks.as_strided is better.
    # return np.lib.stride_tricks.as_strided(
    #     x, shape=shape, strides=strides)
    try:
        if xp == np:
            x = np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
        else:
            x = x.view()
            x._set_shape_and_strides(strides=strides, shape=shape)
        # return np.ndarray.__new__(np.ndarray, strides=strides,
        #                           shape=shape, buffer=x, dtype=x.dtype)
    except Exception:
        print('strides:', x.strides, ' -> ', strides)
        print('shape:', x.shape, ' -> ', shape)
        print('flags:', x.flags)
        print('Parameters:')
        print('shift:', shift, 'Note: negative shift is implemented with a '
                               'following flip')
        print('length:', length, '<- Has to be positive.')
        raise
    if do_flip:
        return xp.flip(x, axis=axis)
    else:
        return x


"""
From http://wiki.scipy.org/Cookbook/SegmentAxis
"""


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """ Generate a new array that chops the given array along the given axis into overlapping frames.

    :param a: The array to segment
    :param length: The length of each frame
    :param overlap: The number of array elements by which the frames should overlap
    :param axis: The axis to operate on; if None, act on the flattened array
    :param end: What to do with the last frame, if the array is not evenly
        divisible into pieces. Options are:
        * 'cut'   Simply discard the extra values
        * 'wrap'  Copy values from the beginning of the array
        * 'pad'   Pad with a constant value
    :param endvalue: The value to use for end='pad'
    :return:

    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').

    Example
    -------
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    >>> segment_axis(np.arange(5).reshape(5), 4, 3, axis=0)
    array([[0, 1, 2, 3],
           [1, 2, 3, 4]])
    >>> segment_axis(np.arange(10).reshape(2, 5), 4, 3, axis=-1)
    array([[[0, 1, 2, 3],
            [1, 2, 3, 4]],
    <BLANKLINE>
           [[5, 6, 7, 8],
            [6, 7, 8, 9]]])
    >>> segment_axis(np.arange(10).reshape(5, 2).T, 4, 3, axis=1)
    array([[[0, 2, 4, 6],
            [2, 4, 6, 8]],
    <BLANKLINE>
           [[1, 3, 5, 7],
            [3, 5, 7, 9]]])
    >>> a = np.arange(5).reshape(5)
    >>> b = segment_axis(a, 4, 2, axis=0)
    >>> a += 1  # a and b point to the same memory
    >>> b
    array([[1, 2, 3, 4]])
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError(
            "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError(
            "overlap must be nonnegative and length must be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (
                length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (
                length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (
            roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    l = a.shape[axis]
    if l == 0:
        raise ValueError(
            "Not enough data points to segment array in 'cut' mode; "
            "try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)

    axis = axis % a.ndim  # force axis >= 0

    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
        axis + 1:]
    if not a.flags.contiguous:
        a = a.copy()
        s = a.strides[axis]
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
            axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError or ValueError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
            axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def to_ndarray(data, copy=True):
    if copy:
        cp = lambda x: np.copy(x)
    else:
        cp = lambda x: x
    if str(type(data)) == "<class 'chainer.variable.Variable'>":
        return cp(data.num)
    elif isinstance(data, np.ndarray):
        return cp(data)
    elif isinstance(data, numbers.Number):
        return data
    elif isinstance(data, collections.Iterable):
        return np.asarray(data)
    else:
        raise ValueError('Unknown type of data {}. Cannot add to list'
                         .format(type(data)))


def stack_context(X, left_context=0, right_context=0, step_width=1):
    """ Stack TxBxF format with left and right context.

    There is a notebook, which illustrates this feature with many details in
    the example notebooks repository.

    :param X: Data with TxBxF format.
    :param left_context: Length of left context.
    :param right_context: Length of right context.
    :param step_width: Step width.
    :return: Stacked features with symmetric padding and head and tail.
    """
    X_stacked = tbf_to_tbchw(
        X,
        left_context=left_context,
        right_context=right_context,
        step_width=step_width
    )[:, :, 0, :].transpose((0, 1, 3, 2))

    T, B, F, W = X_stacked.shape
    X_stacked = X_stacked.reshape(T, B, F * W)

    return X_stacked


def unstack_context(X, mode, left_context=0, right_context=0, step_width=1):
    """ Unstacks stacked features.

    This only works in special cases. Right now, only mode='center'
    is supported. It will return just the center frame and drop the remaining
    parts.

    Other options are related to combining overlapping context frames.

    :param X: Stacked features (or output of your network)
    :param X: mode
    :param left_context: Length of left context.
    :param right_context: Length of right context.
    :param step_width: Step width.
    :return: Data with TxBxF format.
    """

    assert step_width == 1
    context_length = left_context + 1 + right_context
    assert X.shape[2] % context_length == 0
    F = X.shape[2] // context_length

    if mode == 'center':
        return X[:, :, left_context * F:(left_context + 1) * F]
    else:
        NotImplementedError(
            'All other unstack methods are not yet implemented.'
        )


def split_complex_features(X):
    """ Split a complex valued input array into two stacked real parts.

    :param variable: Complex input array with T times B times F features
    :return: Real output array with T times B times 2*F features
    """
    return np.concatenate((np.asarray(X.real), np.asarray(X.imag)), axis=2)


def merge_complex_features(X):
    """ Merge a two stacked real parts into a complex array.

    :param variable: Real input array with T times B times 2*F features
    :return: Complex input array with T times B times F features
    """
    bins = X.shape[-1]
    return X[:, :, :bins // 2] + 1j * X[:, :, bins // 2:]


def tbf_to_tbchw(x, left_context, right_context, step_width,
                 pad_mode='symmetric', pad_kwargs=None):
    """ Transfroms data from TxBxF format to TxBxCxHxW format

    This is only relevant for training a neural network in frames mode.

    The abbreviations stand for:

    T: Time frames
    B: Batch size
    F: Feature size
    C: Channel (almost always 1)
    H: Height of the convolution filter
    W: Width of the convolution filter

    :param x: Data to be transformed
    :param left_context: Context size left to current frame
    :param right_context: Context size right to current frame
    :param step_width: Step width for window
    :param pad_mode: Mode for padding. See :numpy.pad for details
    :param pad_kwargs: Kwargs for pad call
    :return: Transformed data
    """
    if pad_kwargs is None:
        pad_kwargs = dict()
    x = np.pad(x,
               ((left_context, right_context), (0, 0), (0, 0)),
               mode=pad_mode, **pad_kwargs)
    window_size = left_context + right_context + 1
    return segment_axis(
        x, window_size, window_size - step_width,
        axis=0, end='cut').transpose(0, 2, 3, 1)[:, :, None, :, :]


def pad_to(array, to, constant_value=0):
    """ One dimensional padding with zeros to the size of the target array

    :param array: Input array which will be part of the result
    :param to: Target array. Its size will be used to determine the size of the
        return array.
    :return: Padded array
    """
    array = np.array(array)
    result = constant_value * np.ones((len(to),), dtype=array.dtype)
    result[:array.shape[0]] = array
    return result


def pad_axis(array, pad_width, *, axis, mode='constant', **pad_kwargs):
    """ Wrapper around np.pad to support the axis argument.
    This function has mode='constant' as default.

    >>> pad_axis(np.ones([3, 4]), 1, axis=0)
    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [0., 0., 0., 0.]])
    >>> pad_axis(np.ones([3, 4]), 1, axis=1)
    array([[0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.]])
    >>> pad_axis(np.ones([3, 4]), (0, 1), axis=1)
    array([[1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.]])
    >>> pad_axis(np.ones([3, 4]), (1, 0), axis=1)
    array([[0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.]])

    Since np.pad has no axis argument the behaviour for
    isinstance(pad_width, int) is rarely the desired behaviour:

    >>> np.pad(np.ones([3, 4]), 1, mode='constant')
    array([[0., 0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0.]])

    Here the corresponding np.pad calls for above examples:

    >>> np.pad(np.ones([3, 4]), ((1,), (0,)), mode='constant')
    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [0., 0., 0., 0.]])
    >>> np.pad(np.ones([3, 4]), ((0,), (1,)), mode='constant')
    array([[0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.]])
    >>> np.pad(np.ones([3, 4]), ((0, 0), (0, 1)), mode='constant')
    array([[1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.]])
    >>> np.pad(np.ones([3, 4]), ((0, 0), (1, 0)), mode='constant')
    array([[0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.]])


    """
    array = np.asarray(array)

    npad = np.zeros([array.ndim, 2], dtype=np.int)
    npad[axis, :] = pad_width
    return np.pad(array, pad_width=npad, mode=mode, **pad_kwargs)


def _normalize(op):
    op = op.replace(',', '')
    op = op.replace(' ', '')
    op = ' '.join(c for c in op)
    op = op.replace(' * ', '*')
    op = op.replace('- >', '->')
    op = op.replace('. . .', '...')
    return op


def _shrinking_reshape(array, source, target):
    source, target = source.split(), target.replace(' * ', '*').split()

    if '...' in source:
        assert '...' in target, (source, target)
        independent_dims = array.ndim - len(source) + 1
        import string
        ascii_letters = [
            s
            for s in string.ascii_letters
            if s not in source and s not in target
        ]
        index = source.index('...')
        source[index:index + 1] = ascii_letters[:independent_dims]
        index = target.index('...')
        target[index:index + 1] = ascii_letters[:independent_dims]

    input_shape = {key: array.shape[index] for index, key in enumerate(source)}

    output_shape = []
    for t in target:
        product = 1
        if not t == '1':
            t = t.split('*')
            for t_ in t:
                product *= input_shape[t_]
        output_shape.append(product)

    return array.reshape(output_shape)


def _expanding_reshape(array, source, target, **shape_hints):

    try:  # Check number of inputs for unflatten operations
        assert len(re.sub(r'.\*', '', source.replace(' ', ''))) == array.ndim, \
            (array.shape, source, target)
    except AssertionError:  # Check number of inputs for ellipses operations
        assert len(re.sub(r'(\.\.\.)|(.\*)', '', source.replace(' ', ''))) <= \
               array.ndim,(array.shape, source, target)
        

    def _get_source_grouping(source):
        """
        Gets axis as alphanumeric.
        """

        source = ' '.join(source)
        source = source.replace(' * ', '*')
        groups = source.split()
        groups = [group.split('*') for group in groups]
        return groups

    if '*' not in source:
        return array

    source, target = source.split(), target.replace(' * ', '*').split()

    if '...' in source:
        assert '...' in target, (source, target)
        independent_dims = array.ndim - len(source) + 1
        import string
        ascii_letters = [
            s
            for s in string.ascii_letters
            if s not in source and s not in target
        ]
        index = source.index('...')
        source[index:index + 1] = ascii_letters[:independent_dims]
        index = target.index('...')
        target[index:index + 1] = ascii_letters[:independent_dims]

    target_shape = []

    for axis, group in enumerate(_get_source_grouping(source)):
        if len(group) == 1:
            target_shape.append(array.shape[axis:axis + 1])
        else:
            shape_wildcard_remaining = True
            for member in group:
                if member in shape_hints:
                    target_shape.append([shape_hints[member]])
                else:
                    if shape_wildcard_remaining:
                        shape_wildcard_remaining = False
                        target_shape.append([-1])
                    else:
                        raise ValueError('Not enough shape hints provided.')

    target_shape = np.concatenate(target_shape, 0)
    array = array.reshape(target_shape)
    return array


def morph(operation, array, reduce=None, **shape_hints):
    """ This is an experimental version of a generalized reshape.
    See test cases for examples.
    """
    operation = _normalize(operation)
    source, target = operation.split('->')

    # Expanding reshape
    array = _expanding_reshape(array, source, target, **shape_hints)

    # Initial squeeze
    squeeze_operation = operation.split('->')[0].split()
    for axis, op in reversed(list(enumerate(squeeze_operation))):
        if op == '1':
            array = np.squeeze(array, axis=axis)

    # Transpose
    transposition_operation = operation.replace('1', ' ').replace('*', ' ')
    try:
        in_shape, out_shape, (array, ) = _parse_einsum_input([transposition_operation.replace(' ', ''), array])

        if len(set(in_shape) - set(out_shape)) > 0:
            assert reduce is not None, ('Missing reduce function', reduce, transposition_operation)

            reduce_axis = tuple([i for i, s in enumerate(in_shape) if s not in out_shape])
            array = reduce(array, axis=reduce_axis)
            in_shape = ''.join([s for s in in_shape if s in out_shape])

        array = np.einsum(f'{in_shape}->{out_shape}', array)
    except ValueError as e:
        msg = (
            f'op: {transposition_operation} ({in_shape}->{out_shape}), '
            f'shape: {np.shape(array)}'
        )

        if len(e.args) == 1:
            e.args = (e.args[0] + '\n\n' + msg,)
        else:
            print(msg)
        raise

    # Final reshape
    source = transposition_operation.split('->')[-1]
    target = operation.split('->')[-1]

    return _shrinking_reshape(array, source, target)


def add_context(data, left_context=0, right_context=0, step=1,
                cnn_features=False, deltas_as_channel=False,
                num_deltas=2, sequence_output=True):
    if cnn_features:
        data = tbf_to_tbchw(data, left_context, right_context, step,
                            pad_mode='constant',
                            pad_kwargs=dict(constant_values=(0,)))
        if deltas_as_channel:
            feature_size = data.shape[3] // (1 + num_deltas)
            data = np.concatenate(
                [data[:, :, :, i * feature_size:(i + 1) * feature_size, :]
                 for i in range(1 + num_deltas)], axis=2)
    else:
        data = stack_context(data, left_context=left_context,
                             right_context=right_context, step_width=step)
        if not sequence_output:
            data = np.concatenate(
                [data[:, i, ...].reshape((-1, data.shape[-1])) for
                 i in range(data.shape[1])], axis=0)
    return data


# http://stackoverflow.com/a/3153267
def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros),
                             axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def labels_to_one_hot(
        labels: np.ndarray, categories: int, axis: int = 0,
        keepdims=False, dtype=np.float32
):
    """ Translates an arbitrary ndarray with labels to one hot coded array.

    Args:
        labels: Array with any shape and integer labels.
        categories: Maximum integer label larger or equal to maximum of the
            labels ndarray.
        axis: Axis along which the one-hot vector will be aligned.
        keepdims:
            If keepdims is True, this function behaves similar to
            numpy.concatenate(). It will expand the provided axis.
            If keepdims is False, it will create a new axis along which the
            one-hot vector will be placed.
        dtype: Provides the dtype of the output one-hot mask.

    Returns:
        One-hot encoding with shape (..., categories, ...).

    """
    if keepdims:
        assert labels.shape[axis] == 1
        result_ndim = labels.ndim
    else:
        result_ndim = labels.ndim + 1

    if axis < 0:
        axis += result_ndim

    shape = labels.shape
    zeros = np.zeros((categories, labels.size), dtype=dtype)
    zeros[labels.ravel(), range(labels.size)] = 1

    zeros = zeros.reshape((categories,) + shape)

    if keepdims:
        zeros = zeros[(slice(None),) * (axis + 1) + (0,)]

    zeros = np.moveaxis(zeros, 0, axis)

    return zeros


@dataclass
class Cutter:
    """
    Implements cut and expand for low_cut and high_cut. Often interesting when
    you want to avoid processing of some frequencies when beamforming.

    Why do we enforce negative upper end: Positive values can be confusing. You
    may want to cut `n` values or want to keep up to `n`.

    >>> c = Cutter(1, -2)
    >>> array = np.array([[1, 2, 3, 4]])
    >>> c.cut(array, axis=1)
    array([[2]])

    >>> c.expand(c.cut(array, axis=1), axis=1)
    array([[0, 2, 0, 0]])

    >>> c.overwrite(array, axis=1)
    array([[0, 2, 0, 0]])
    """
    low_cut: int
    high_cut: int

    def __post_init__(self):
        assert self.low_cut >= 0, 'Positive or zero starting index'
        assert self.high_cut <= 0, 'Negative or zero ending index'

    def cut(self, array, *, axis):
        """Cuts start and end."""
        assert isinstance(axis, int), axis
        trimmer = [slice(None)] * array.ndim
        trimmer[axis] = slice(self.low_cut, self.high_cut)
        return array[trimmer]

    def expand(self, array, *, axis):
        """Pads to reverse the cut."""
        assert isinstance(axis, int), axis
        return pad_axis(array, (self.low_cut, -self.high_cut), axis=axis)

    def overwrite(self, array, *, axis):
        """Returns a copy with start end end filled with zeros."""
        return self.expand(self.cut(array, axis=axis), axis=axis)
