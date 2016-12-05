"""
This module deals with all sorts of audio input and output.
"""
import librosa
import wave
from io import BytesIO
import inspect
from os import path, remove
import tempfile
import nt.utils.process_caller as pc
import numpy as np
import glob
from pathlib import Path


UTILS_DIR = path.join(path.dirname(path.abspath(
        inspect.getfile(inspect.currentframe()))), 'utils')


def audioread(path, offset=0.0, duration=None, sample_rate=16000):
    """
    Reads a wav file, converts it to 32 bit float values and reshapes according
    to the number of channels.
    Now, this is a wrapper of librosa with our common defaults.

    :param path: Absolute or relative file path to audio file.
    :type: String.
    :param offset: Begin of loaded audio.
    :type: Scalar in seconds.
    :param duration: Duration of loaded audio.
    :type: Scalar in seconds.
    :param sample_rate: Resamples utterances to given value. None = native
        sample rate.
    :type: scalar in number of samples per second
    :return:

    .. admonition:: Example:
        Only path provided:

        >>> path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        >>> signal = audioread(path)

        Say you load audio examples from a very long audio, you can provide a
        start position and a duration in seconds.

        >>> path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        >>> signal = audioread(path, offset=0, duration=1)
    """

    if isinstance(path, Path):
        path = str(path)

    signal = librosa.load(path,
                          sr=sample_rate,
                          mono=False,
                          offset=offset,
                          duration=duration)
    return signal[0]


def read_nist_wsj(path, sample_rate=16000):
    """
    Converts a nist/sphere file of wsj and reads it with audioread.

    :param path: file path to audio file.
    :param sample_rate: Resamples utterances to given value. None = native
        sample rate.
    :return:
    """
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    cmd = "{}/sph2pipe -f wav {path} {dest_file}".format(
        UTILS_DIR, path=path, dest_file=tmp_file.name)
    dir = "{}/sph2pipe".format(UTILS_DIR)
    # subprocess.Popen([dir , '-f', 'wav', path, tmp_file.name])
    pc.run_processes(cmd, ignore_return_code=False)
    signal = audioread(tmp_file.name, sample_rate=sample_rate)
    remove(tmp_file.name)
    return signal


def read_raw(path, dtype=np.dtype('<i2')):
    """
    Reads raw data (tidigits data)

    :param path: file path to audio file
    :param dtype: datatype, default: int16, little-endian
    :return: numpy array with audio samples
    """

    if isinstance(path, Path):
        path = str(path)

    with open(path, 'rb') as f:
        return np.fromfile(f, dtype=dtype)


def getparams(path):
    """
    Returns parameters of wav file.

    :param path: Absolute or relative file path to audio file.
    :type: String.
    :return: Named tuple with attributes (nchannels, sampwidth, framerate,
    nframes, comptype, compname)
    """
    with wave.open(path, 'r') as wave_file:
        return wave_file.getparams()


def read_from_byte_string(byte_string, dtype=np.dtype('<i2')):
    """ Parses a bytes string, i.e. a raw read of a wav file

    :param byte_string: input bytes string
    :param dtype: dtype used to decode the audio data
    :return: np.ndarray with audio data with channels x samples
    """
    wav_file = wave.openfp(BytesIO(byte_string))
    channels = wav_file.getnchannels()
    interleaved_audio_data = np.frombuffer(
        wav_file.readframes(wav_file.getnframes()), dtype=dtype)
    audio_data = np.array([interleaved_audio_data[ch::channels] for ch in range(channels)])
    audio_data = audio_data.astype(np.float32) / np.max(audio_data)
    return audio_data