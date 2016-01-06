"""
This file contains the STFT function and related helper functions.
"""
import numpy
from math import ceil
import scipy

from scipy import signal
from numpy.fft import rfft, irfft

import string
from nt.utils.numpy_utils import segment_axis


def stft(time_signal, time_dim=None, size=1024, shift=256,
         window=signal.blackman, fading=True, window_length=None):
    """
    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.

    :param time_signal: multi channel time signal.
    :param time_dim: Scalar dim of time.
        Default: None means the biggest dimension
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    if time_dim is None:
        time_dim = numpy.argmax(time_signal.shape)

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        pad = [(0, 0)]*time_signal.ndim
        pad[time_dim] = [size-shift, size-shift]
        time_signal = numpy.pad(time_signal, pad, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(time_signal.shape[time_dim], size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    pad = [(0, 0)]*time_signal.ndim
    pad[time_dim] = [0, samples - time_signal.shape[time_dim]]
    time_signal = numpy.pad(time_signal, pad, mode='constant')

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = numpy.pad(window, (0, size-window_length), mode='constant')

    time_signal_seg = segment_axis(time_signal, size,
                                   size-shift, axis=time_dim)

    letters = string.ascii_lowercase
    mapping = letters[:time_signal_seg.ndim] + ',' + letters[time_dim + 1] \
        + '->' + letters[:time_signal_seg.ndim]

    # ToDo: Implement this more memory efficient
    return rfft(numpy.einsum(mapping, time_signal_seg, window),
                axis=time_dim + 1)


def stft_single_channel(time_signal, size=1024, shift=256,
         window=signal.blackman, fading=True, window_length=None):
    """
    Calculates the short time Fourier transformation of a single channel time
    signal. It is able to add additional zeros for fade-in and fade out and
    should yield an STFT signal which allows perfect reconstruction.

    Up to now, only a single channel time signal is possible.

    :param time_signal: Single channel time signal.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    assert len(time_signal.shape) == 1

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        time_signal = numpy.pad(time_signal, size-shift, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(len(time_signal), size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    time_signal = numpy.pad(time_signal,
                         (0, samples - len(time_signal)), mode='constant')

    # The range object contains the sample index
    # of the beginning of each frame.
    range_object = range(0, len(time_signal) - size + shift, shift)

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = numpy.pad(window, (0, size-window_length), mode='constant')
    windowed = numpy.array([(window*time_signal[i:i+size])
                            for i in range_object])
    return rfft(windowed)


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
    Matlab impelementation in terms of variable names.

    The results are equal.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert numpy.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = numpy.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts+1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] \
                    += analysis_window[analysis_index] ** 2

    sum_of_squares = numpy.kron(numpy.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
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
    assert numpy.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = numpy.zeros(shift)
    for synthesis_index in range(0, shift):
        sample_index = numpy.arange(0, number_of_shifts+1)
        analysis_index = synthesis_index + sample_index * shift
        analysis_index = analysis_index[analysis_index + 1 < fft_size]
        sum_of_squares[synthesis_index] \
            = numpy.sum(analysis_window[analysis_index] ** 2)
    sum_of_squares = numpy.kron(numpy.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window

def istft_loop(stft_signal, time_dim=-2, freq_dim=-1):

    def convert_for_mat_loopy(tensor, mat_dim_one, mat_dim_two):
        ndim = tensor.ndim
        shape = tensor.shape
        mat_dim_one %= ndim
        mat_dim_two %= ndim
        perm = [x for x in range(ndim)
                if x not in (mat_dim_one, mat_dim_two)] \
            + [mat_dim_one, mat_dim_two]
        tensor = tensor.transpose(perm)
        return numpy.reshape(tensor,
                             (-1, shape[mat_dim_one], shape[mat_dim_two]))

    def reconstruct_mat_loopy(tensor, dim_one, dim_two, shape):
        ndim = len(shape)
        dim_one %= ndim
        dim_two %= ndim
        new_shape = [shape[x] for x in range(ndim)
                     if x not in (dim_one, dim_two)] \
            + [shape[dim_one], shape[dim_two]]
        tensor = numpy.reshape(tensor, new_shape)
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

    time_signal = numpy.array([istft(stft_signal[i, :, :])
                               for i in range(stft_signal.shape[0])])
    shape = list(shape)
    shape[time_dim] = 1
    shape[freq_dim] = time_signal.shape[1]
    time_signal = reconstruct_mat_loopy(
        numpy.expand_dims(time_signal, axis=-2), time_dim, freq_dim, shape)
    return numpy.squeeze(time_signal, axis=time_dim)

def istft(stft_signal, size=1024, shift=256,
          window=signal.blackman, fading=True, window_length=None):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

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
    assert stft_signal.shape[1] == size // 2 + 1

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = numpy.pad(window, (0, size-window_length), mode='constant')

    window = _biorthogonal_window_loopy(window, shift)

    # Why? Line created by Hai, Lukas does not know, why it exists.
    window *= size

    time_signal = scipy.zeros(stft_signal.shape[0] * shift + size - shift)

    for j, i in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[i:i + size] += window * numpy.real(irfft(stft_signal[j]))

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[size-shift:len(time_signal)-(size-shift)]

    return time_signal


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal.
    The output is guaranteed to be real.

    :param stft: Complex STFT signal with dimensions
        #time_frames times #frequency_bins.
    :return: Real spectrogram with same dimensions as input.
    """
    spectrogram = numpy.abs(stft_signal * numpy.conjugate(stft_signal))
    return spectrogram


def spectrogram_to_energy_per_frame(spectrogram):
    """
    The energy per frame is sometimes used as an additional feature to the MFCC
    features. Here, it is calculated from the power spectrum.

    :param spectrogram: Real valued power spectrum.
    :return: Real valued energy per frame.
    """
    energy = numpy.sum(spectrogram, 1)

    # If energy is zero, we get problems with log
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)
    return energy


def get_stft_center_frequencies(size=1024, sample_rate=16000):
    """
    It is often necessary to know, which center frequency is
    represented by each frequency bin index.

    :param size: Scalar FFT-size.
    :param sample_rate: Scalar sample frequency in Hertz.
    :return: Array of all relevant center frequencies
    """
    frequency_index = numpy.arange(0, size/2 + 1)
    return frequency_index * sample_rate / size
