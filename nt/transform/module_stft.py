"""
This file contains the STFT function and related helper functions.
"""
import typing
import string
from math import ceil

import numpy as np
from numpy.fft import rfft, irfft
from scipy import signal

from nt.utils.numpy_utils import segment_axis_v2
from nt.utils.numpy_utils import roll_zeropad


def stft(
        time_signal,
        size: int=1024,
        shift: int=256,
        *,
        axis=-1,
        window: typing.Callable=signal.blackman,
        window_length: int=None,
        fading: bool=True,
        pad: bool=True,
        symmetric_window: bool=False,
) -> np.array:
    """ !!! WIP !!!
    ToDo: Open points:
     - sym_window need literature
     - fading why it is better?
     - should pad have more degrees of freedom?

    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.

    :param time_signal: Multi channel time signal with dimensions
        AA x ... x AZ x T x BA x ... x BZ.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift, the step between successive frames in
        samples. Typically shift is a fraction of size.
    :param axis: Scalar axis of time.
        Default: None means the biggest dimension.
    :param window: Window function handle. Default is blackman window.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :param pad: If true zero pad the signal to match the shape, else cut
    :param symmetric_window: symmetric or periotic window. Assume window is periodic.
        Since the implementation of the windows in scipy.signal have a curious
        behaviour for odd window_length. Use window(len+1)[:-1]. Since is equal
        to the behaviour of MATLAB.
    :return: Single channel complex STFT signal with dimensions
        AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    time_signal = np.array(time_signal)

    axis = axis % time_signal.ndim

    if window_length is None:
        window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int)
        pad_width[axis, :] = window_length - shift
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    if symmetric_window:
        window = window(window_length)
    else:
        # https://github.com/scipy/scipy/issues/4551
        window = window(window_length + 1)[:-1]

    time_signal_seg = segment_axis_v2(time_signal, window_length,
                                      shift=shift, axis=axis,
                                      end='pad' if pad else 'cut')

    letters = string.ascii_lowercase[:time_signal_seg.ndim]
    mapping = letters + ',' + letters[axis + 1] + '->' + letters

    try:
        # ToDo: Implement this more memory efficient
        return rfft(np.einsum(mapping, time_signal_seg, window),
                    n=size, axis=axis + 1)
    except ValueError as e:
        raise ValueError(
            f'Could not calculate the stft, something does not match.\n'
            f'mapping: {mapping}, '
            f'time_signal_seg.shape: {time_signal_seg.shape}, '
            f'window.shape: {window.shape}, '
            f'size: {size}'
            f'axis+1: {axis+1}'
        ) from e


def stft_with_kaldi_dimensions(
        time_signal,
        size: int = 512,
        shift: int = 160,
        *,
        window=signal.blackman,
        window_length=400,
        symmetric_window: bool = False
):
    """
    The Kaldi implementation uses another non standard window.
    See:
     - https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h#L48
     - https://github.com/kaldi-asr/kaldi/blob/81b7a1947fb8df501a2bbb680b65ce18ce606cff/src/feat/feature-window.h#L48

    ..note::
       Kaldi uses symmetric_window == True
        - https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.cc#L113

    """
    # ToDo: Write test to force this function to fulfill the old kaldi_dims
    #       argument
    # if kaldi_dims:
    #     nsamp = time_signal.shape[axis]
    #     frames = time_signal_seg.shape[axis]
    #     expected_frames = 1 + ((nsamp - size) // shift)
    #     if frames != expected_frames:
    #         raise ValueError('Expected {} frames, got {}'.format(
    #             expected_frames, frames))
    return stft(
        time_signal,
        size=size,
        shift=shift,
        window=window,
        window_length=window_length,
        fading=False,
        pad=False,
        symmetric_window=symmetric_window
    )


def _samples_to_stft_frames(samples, size, shift):
    """
    Calculates STFT frames from samples in time domain.
    :param samples: Number of samples in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of STFT frames.

    >>> _samples_to_stft_frames(19, 16, 4)
    2
    >>> _samples_to_stft_frames(20, 16, 4)
    2
    >>> _samples_to_stft_frames(21, 16, 4)
    3
    """
    # I changed this from np.ceil to math.ceil, to yield an integer result.
    return ceil((samples - size + shift) / shift)


def _stft_frames_to_samples(frames, size, shift):
    """
    Calculates samples in time domain from STFT frames
    :param frames: Number of STFT frames.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of samples in time domain.

    >>> _stft_frames_to_samples(2, 16, 4)
    20
    """
    return frames * shift + size - shift


def _biorthogonal_window_loopy(analysis_window, shift):
    """
    This version of the synthesis calculation is as close as possible to the
    Matlab implementation in terms of variable names.

    The results are equal.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts+1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] \
                    += analysis_window[analysis_index] ** 2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size

    # Why? Line created by Hai, Lukas does not know, why it exists.
    synthesis_window *= fft_size

    return synthesis_window


def _biorthogonal_window(analysis_window, shift):
    """
    This is a vectorized implementation of the window calculation. It is much
    slower than the variant using for loops.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        sample_index = np.arange(0, number_of_shifts+1)
        analysis_index = synthesis_index + sample_index * shift
        analysis_index = analysis_index[analysis_index + 1 < fft_size]
        sum_of_squares[synthesis_index] \
            = np.sum(analysis_window[analysis_index] ** 2)
    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size

    # Why? Line created by Hai, Lukas does not know, why it exists.
    synthesis_window *= fft_size

    return synthesis_window


def _biorthogonal_window_brute_force(analysis_window, shift,
                                     use_amplitude=False):
    """
    The biorthogonal window (synthesis_window) must verify the criterion:
        synthesis_window * analysis_window plus it's shifts must be one.
        1 == sum m from -inf to inf over (synthesis_window(n - mB) * analysis_window(n - mB))
        B ... shift
        n ... time index
        m ... shift index

    :param analysis_window:
    :param shift:
    :return:

    >>> analysis_window = signal.blackman(4+1)[:-1]
    >>> print(analysis_window)
    [ -1.38777878e-17   3.40000000e-01   1.00000000e+00   3.40000000e-01]
    >>> synthesis_window = _biorthogonal_window_brute_force(analysis_window, 1)
    >>> print(synthesis_window)
    [ -1.12717575e-17   2.76153346e-01   8.12215724e-01   2.76153346e-01]
    >>> mult = analysis_window * synthesis_window
    >>> sum(mult)
    1.0000000000000002
    """
    size = len(analysis_window)

    influence_width = (size - 1) // shift

    denominator = np.zeros_like(analysis_window)

    if use_amplitude:
        analysis_window_square = analysis_window
    else:
        analysis_window_square = analysis_window ** 2
    for i in range(-influence_width, influence_width + 1):
        denominator += roll_zeropad(analysis_window_square, shift * i)

    if use_amplitude:
        synthesis_window = 1 / denominator
    else:
        synthesis_window = analysis_window / denominator
    return synthesis_window


_biorthogonal_window_fastest = _biorthogonal_window_brute_force


def istft(
        stft_signal,
        size: int=1024,
        shift: int=256,
        *,
        window: typing.Callable=signal.blackman,
        fading: bool=True,
        window_length: int=None,
        symmetric_window: bool=False,
        # use_amplitude_for_biorthogonal_window=False,
        # disable_sythesis_window=False
):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    ..note::
        Be careful if you make modifications in the frequency domain (e.g.
        beamforming) because the synthesis window is calculated according to
        the unmodified! analysis window.

    :param stft_signal: Single channel complex STFT signal
        with dimensions (..., frames, size/2+1).
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Removes the additional padding, if done during STFT.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
    :return: Single channel time signal.
    """
    # Note: frame_axis and frequency_axis would make this function much more
    #       complicated
    stft_signal = np.array(stft_signal)

    assert stft_signal.shape[-1] == size // 2 + 1, str(stft_signal.shape)

    if window_length is None:
        window_length = size

    if symmetric_window:
        window = window(window_length)
    else:
        window = window(window_length + 1)[:-1]

    window = _biorthogonal_window_fastest(window, shift)

    # window = _biorthogonal_window_fastest(
    #     window, shift, use_amplitude_for_biorthogonal_window)
    # if disable_sythesis_window:
    #     window = np.ones_like(window)

    time_signal = np.zeros(
        (*stft_signal.shape[:-2],
         stft_signal.shape[-2] * shift + window_length - shift))

    # Get the correct view to time_signal
    time_signal_seg = segment_axis_v2(
        time_signal, window_length, shift, end=None)

    # Unbuffered inplace add
    np.add.at(time_signal_seg, ...,
              window * np.real(irfft(stft_signal))[..., :window_length])
    # The [..., :window_length] is the inverse of the window padding in rfft.

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[
            ..., window_length - shift:time_signal.shape[-1] - (window_length - shift)]

    return time_signal


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal.
    The output is guaranteed to be real.

    :param stft_signal: Complex STFT signal with dimensions
        #time_frames times #frequency_bins.
    :return: Real spectrogram with same dimensions as input.

    Note: Special version of nt.math.misc.abs_square
    """

    spectrogram = stft_signal.real**2 + stft_signal.imag**2
    return spectrogram


def spectrogram(time_signal, *args, **kwargs):
    """ Thin wrapper of stft with power spectrum calculation.

    :param time_signal:
    :param args:
    :param kwargs:
    :return:
    """
    return stft_to_spectrogram(stft(time_signal, *args, **kwargs))


def spectrogram_to_energy_per_frame(spectrogram):
    """
    The energy per frame is sometimes used as an additional feature to the MFCC
    features. Here, it is calculated from the power spectrum.

    :param spectrogram: Real valued power spectrum.
    :return: Real valued energy per frame.
    """
    energy = np.sum(spectrogram, 1)

    # If energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    return energy


def get_stft_center_frequencies(size=1024, sample_rate=16000):
    """
    It is often necessary to know, which center frequency is
    represented by each frequency bin index.

    :param size: Scalar FFT-size.
    :param sample_rate: Scalar sample frequency in Hertz.
    :return: Array of all relevant center frequencies
    """
    frequency_index = np.arange(0, size/2 + 1)
    return frequency_index * sample_rate / size
