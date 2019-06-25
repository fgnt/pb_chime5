"""
All provided masking functions expect the complex valued stft signal as input.
Each masking function should not take care of further convenience functions
than allowing arbitrary sensor_axis and any number of independent dimensions.

Only, when a multichannel signal is used to pool the power along channels,
the sensor_axis can be provided.

All other convenience should be dealt with from wrapper functions which possibly
take the masking function as a callback. If you want to use lists of arrays,
write an appropriate wrapper function.

If desired, concatenation of *ins can be done in a decorator.

When appropriate, functions assume that the target speaker is channel 0 and
noise is channel 1.

Optional axis parameters are:
 * ``source_axis`` with default ``0``.
 * ``sensor_axis`` with default ``None``. If given, it is used for pooling.
 * ``frequency_axis`` with default ``-2``.
 * ``time_axis`` with default ``-1``.

All other axes are regarded as independent dimensions.
"""

# TODO: Migrate and remove this files:
# TODO:  - tests/speech_enhancement_test/test_merl_masks.py
# TODO:  - nt/speech_enhancement/merl_masks.py
# TODO:  - nt/speech_enhancement/mask_estimation.py
# TODO: Add test-case for LorenzMask
# CB: Eventuell einen Dekorator nutzen für force signal np.ndarray?
# CB: Eventuell einen Dekorator nutzen für force signal.real.dtype == return.dtype?

import numpy as np

EPS = 1e-18


def lorenz_mask(
        signal: np.ndarray,
        *,
        sensor_axis=None,
        axis=(-2, -1),
        lorenz_fraction: float=0.98,
        weight: float=0.999,
) -> np.ndarray:
    """ Calculate softened mask according to Lorenz function criterion.

    To be precise, the lorenz_fraction is not actually a quantile
    although it is in the range [0, 1]. If it was the quantile fraction, it
    would the the fraction of the number of observations.

    Args:
        signal: Complex valued stft signal.
        sensor_axis:
        axis: time_axis and/or frequency_axis
        lorenz_fraction: Fraction of observations which are rated down
        weight: Governs the influence of the mask

    Returns:

    """
    signal = np.asarray(signal)

    power = np.abs(signal)**2
    if sensor_axis is not None:
        power = power.sum(axis=sensor_axis, keepdims=True)

    if not isinstance(axis, (tuple, list)):
        axis = (axis,)

    # Only works, when last two dimensions are frequency and time.
    tmp_axis = tuple([-i - 1 for i in range(len(axis))])

    power = np.moveaxis(power, axis, tmp_axis)
    shape = power.shape
    working_shape = tuple([
        np.prod(shape[:-len(tmp_axis)], dtype=np.int64),
        np.prod(shape[-len(tmp_axis):]),
    ])

    power = np.reshape(power, working_shape)

    mask = np.zeros_like(power, dtype=power.real.dtype)

    def get_mask(power):
        sorted_power = np.sort(power, axis=None)[::-1]
        lorenz_function = np.cumsum(sorted_power) / np.sum(sorted_power)
        threshold = np.min(sorted_power[lorenz_function < lorenz_fraction])
        _mask = power > threshold
        return _mask

    for i in range(power.shape[0]):
        mask[i, :] = get_mask(power[i])

    mask = 0.5 + weight * (mask - 0.5)

    return np.moveaxis(mask.reshape(shape), tmp_axis, axis)


def quantil_mask(
        signal: np.ndarray,
        quantil=[0.1, -0.9],
        *,
        sensor_axis=None,
        axis=(-2),
        weight: float=0.999,
) -> np.ndarray:
    """

    Args:
        signal:
        quantil: pos for speech, negative for noise
        sensor_axis:
        axis: Suggestion: time axis, Alternative time and frequency axis
        weight:

    Returns:
        Mask of shape [*quantil.shape, *signal.shape]

    """
    signal = np.abs(signal)

    if isinstance(quantil, (tuple, list)):
        return np.array([quantil_mask(signal=signal, sensor_axis=sensor_axis, axis=axis, quantil=q, weight=weight) for q in quantil])

    if sensor_axis is not None:
        signal = signal.sum(axis=sensor_axis, keepdims=True)

    if not isinstance(axis, (tuple, list)):
        axis = (axis,)

    # Convert signal to 2D with [independent, sample axis]
    tmp_axis = tuple([-i - 1 for i in range(len(axis))])
    signal = np.moveaxis(signal, axis, tmp_axis)
    shape = signal.shape
    working_shape = tuple(
        [np.prod(shape[:-len(tmp_axis)]), np.prod(shape[-len(tmp_axis):])])
    signal = np.reshape(signal, working_shape)

    if quantil >= 0:
        threshold = np.percentile(signal, q=(1 - quantil)*100, axis=-1)
    else:
        threshold = np.percentile(signal, q=abs(quantil)*100, axis=-1)

    mask = np.zeros_like(signal)
    for i in range(mask.shape[0]):
        if quantil >= 0:
            mask[i, :] = signal[i, :] > threshold[i]
        else:
            mask[i, :] = signal[i, :] < threshold[i]

    # Drop this line?
    mask = 0.5 + weight * (mask - 0.5)

    # Restore original shape
    mask = np.moveaxis(mask.reshape(shape), tmp_axis, axis)

    if sensor_axis is not None:
        mask = np.squeeze(mask, axis=sensor_axis)
    return mask
