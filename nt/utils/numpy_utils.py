import warnings
import chainer
import numpy as np
import collections
import numbers

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
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length: raise ValueError(
        "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0: raise ValueError(
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
    if l == 0: raise ValueError(
        "Not enough data points to segment array in 'cut' mode; "
        "try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                  axis + 1:]

    if not a.flags.contiguous:
        a = a.copy()
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
    if isinstance(data, chainer.Variable):
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


def _normalize(op):
    op = op.replace(',', '')
    op = op.replace(' ', '')
    op = ' '.join(c for c in op)
    op = op.replace(' * ', '*')
    op = op.replace('- >', '->')
    return op


def _only_reshape(array, source, target):
    source, target = source.split(), target.replace(' * ', '*').split()
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


def reshape(array, operation):
    """ This is an experimental version of a generalized reshape.

    See test cases for examples.
    """
    operation = _normalize(operation)

    if '*' in operation.split('->')[0]:
        raise NotImplementedError(
            'Unflatten operation not supported by design. '
            'Actual values for dimensions are not available to this function.'
        )

    # Initial squeeze
    squeeze_operation = operation.split('->')[0].split()
    for axis, op in reversed(list(enumerate(squeeze_operation))):
        if op == '1':
            array = np.squeeze(array, axis=axis)

    # Transpose
    transposition_operation = operation.replace('1', ' ').replace('*', ' ')
    array = np.einsum(transposition_operation, array)

    # Final reshape
    source = transposition_operation.split('->')[-1]
    target = operation.split('->')[-1]

    return _only_reshape(array, source, target)


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
    if shift == 0: return a
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
