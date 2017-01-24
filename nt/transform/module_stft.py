"""
This file contains the STFT function and related helper functions.
"""
import numpy as np
from math import ceil
import scipy

from scipy import signal
from numpy.fft import rfft, irfft

import string
from nt.utils.numpy_utils import segment_axis, segment_axis_v2
from nt.utils.numpy_utils import roll_zeropad


def stft_v2(
        time_signal,
        size: int=1024,
        shift: int=256,
        axis=None,
        window=signal.blackman,
        window_length=None,
        *,
        fading: bool=True,
        detrend=False,
        pad: bool=True,
        sym_window: bool=True
):
    """ !!! WIP !!!
    ToDo: Discuss new stft:
     - sym_window need literature
     - fading why it is better?
     - use zero padding from rfft instead padding window
     - detrend need literature
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
    :param detrend: Detrends segments before FFT. If ``type == 'linear'``,
        the result of a linear least-squares fit is subtracted.
        If ``type == 'constant'``, only the mean is subtracted. Defaults to False.
    :type detrend: str or bool, optional
    :param pad: If true zero pad the signal to match the shape, else cut
    :param sym_window: symmetric or periotic window. Assume window is periodic.
        Since the implementation of the windows in scipy.signal have a curious
        behaviour for odd window_length. Use window(len+1)[:-1]. Since is equal
        to the behaviour of MATLAB.
    :return: Single channel complex STFT signal with dimensions
        AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    time_signal = np.array(time_signal)

    # TODO: This automatism should be discussed and in my opinion removed (L)
    if axis is None:
        axis = np.argmax(time_signal.shape)

    if window_length is None:
        window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int)
        pad_width[axis, :] = window_length - shift
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    if sym_window:
        # https://github.com/scipy/scipy/issues/4551
        window = window(window_length+1)[:-1]
    else:
        window = window(window_length)

    time_signal_seg = segment_axis_v2(time_signal, window_length,
                                      shift=shift, axis=axis, pad=pad)

    if detrend not in ['linear', 'l', 'constant', 'c', False]:
        raise ValueError("Detrend type must be 'linear', 'constant' or False.")
    elif isinstance(detrend, str):
        time_signal_seg = signal.detrend(time_signal_seg,
                                         type=detrend, axis=axis + 1)

    letters = string.ascii_lowercase[:time_signal_seg.ndim]
    mapping = letters + ',' + letters[axis + 1] + '->' + letters

    # ToDo: Implement this more memory efficient
    return rfft(np.einsum(mapping, time_signal_seg, window),
                n=size, axis=axis + 1)


def stft(
        time_signal, size=1024, shift=256, axis=None, window=signal.blackman,
        fading=True, window_length=None, kaldi_dims=False, detrend=False
):
    """
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
    :param kaldi_dims: No padding or fading.
        Size is 1 + ((nsamp - frame_length) / frame_shift).
    :param detrend: Detrends segments before FFT. If ``type == 'linear'``,
        the result of a linear least-squares fit is subtracted.
        If ``type == 'constant'``, only the mean is subtracted. Defaults to False.
    :type detrend: str or bool, optional
    :return: Single channel complex STFT signal with dimensions
        AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    # TODO: This automatism should be discussed and in my opinion removed (L)
    if axis is None:
        axis = np.argmax(time_signal.shape)

    # Pad with zeros to have enough samples for the window function to fade.
    if fading and not kaldi_dims:
        pad = [(0, 0)]*time_signal.ndim
        pad[axis] = [size - shift, size - shift]
        time_signal = np.pad(time_signal, pad, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    if not kaldi_dims:
        frames = _samples_to_stft_frames(time_signal.shape[axis], size, shift)
        samples = _stft_frames_to_samples(frames, size, shift)
        pad = [(0, 0)]*time_signal.ndim
        pad[axis] = [0, samples - time_signal.shape[axis]]
        time_signal = np.pad(time_signal, pad, mode='constant')

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size-window_length), mode='constant')

    time_signal_seg = segment_axis(time_signal, size, size - shift, axis=axis)
    letters = string.ascii_lowercase
    mapping = letters[:time_signal_seg.ndim] + ',' + letters[axis + 1] \
        + '->' + letters[:time_signal_seg.ndim]

    if detrend not in ['linear', 'l', 'constant', 'c', False]:
        raise ValueError("Detrend type must be 'linear', 'constant' or False.")
    elif isinstance(detrend, str):
        time_signal_seg = signal.detrend(time_signal_seg, type=detrend, axis=axis+1)

    if kaldi_dims:
        nsamp = time_signal.shape[axis]
        frames = time_signal_seg.shape[axis]
        expected_frames = 1 + ((nsamp - size) // shift)
        if frames != expected_frames:
            raise ValueError('Expected {} frames, got {}'.format(
                expected_frames, frames))

    # ToDo: Implement this more memory efficient
    return rfft(np.einsum(mapping, time_signal_seg, window),
                axis=axis + 1)


def _samples_to_stft_frames(samples, size, shift):
    """
    Calculates STFT frames from samples in time domain.
    :param samples: Number of samples in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of STFT frames.
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


def istft_loop(
        stft_signal,
        size=1024, shift=256,
        time_dim=-2, freq_dim=-1,
        window=signal.blackman
):

    def convert_for_mat_loopy(tensor, mat_dim_one, mat_dim_two):
        ndim = tensor.ndim
        shape = tensor.shape
        mat_dim_one %= ndim
        mat_dim_two %= ndim
        perm = [x for x in range(ndim)
                if x not in (mat_dim_one, mat_dim_two)] \
            + [mat_dim_one, mat_dim_two]
        tensor = tensor.transpose(perm)
        return np.reshape(tensor,
                             (-1, shape[mat_dim_one], shape[mat_dim_two]))

    def reconstruct_mat_loopy(tensor, dim_one, dim_two, shape):
        ndim = len(shape)
        dim_one %= ndim
        dim_two %= ndim
        new_shape = [shape[x] for x in range(ndim)
                     if x not in (dim_one, dim_two)] \
            + [shape[dim_one], shape[dim_two]]
        tensor = np.reshape(tensor, new_shape)
        perm = list(range(ndim - 2))
        if dim_one > dim_two:
            perm.insert(dim_two, -1)
            perm.insert(dim_one, -2)
        else:
            perm.insert(dim_one, -2)
            perm.insert(dim_two, -1)
        return tensor.transpose(perm)

    shape = stft_signal.shape
    stft_signal = convert_for_mat_loopy(stft_signal, time_dim, freq_dim)

    time_signal = np.array([istft(
        stft_signal[i, :, :], window=window, size=size, shift=shift
    ) for i in range(stft_signal.shape[0])])
    shape = list(shape)
    shape[time_dim] = 1
    shape[freq_dim] = time_signal.shape[1]
    time_signal = reconstruct_mat_loopy(
        np.expand_dims(time_signal, axis=-2), time_dim, freq_dim, shape)
    return np.squeeze(time_signal, axis=time_dim)


def istft(stft_signal, size=1024, shift=256,
          window=signal.blackman, fading=True, window_length=None,
          use_amplitude_for_biorthogonal_window=False,
          disable_sythesis_window=False):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    ..note::
        Be careful if you make modifications in the frequency domain (e.g.
        beamforming) because the synthesis window is calculated according to the
        unmodified! analysis window.

    :param stft_signal: Single channel complex STFT signal
        with dimensions frames times size/2+1.
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
    assert stft_signal.shape[1] == size // 2 + 1, str(stft_signal.shape)

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size-window_length), mode='constant')
    window = _biorthogonal_window_fastest(window, shift,
                                          use_amplitude_for_biorthogonal_window)
    if disable_sythesis_window:
        window = np.ones_like(window)

    time_signal = scipy.zeros(stft_signal.shape[0] * shift + size - shift)

    for j, i in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[i:i + size] += window * np.real(irfft(stft_signal[j]))

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[size-shift:len(time_signal)-(size-shift)]

    return time_signal


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal.
    The output is guaranteed to be real.

    :param stft_signal: Complex STFT signal with dimensions
        #time_frames times #frequency_bins.
    :return: Real spectrogram with same dimensions as input.
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
