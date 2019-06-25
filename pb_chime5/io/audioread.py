"""
This module deals with all sorts of audio input and output.
"""
import inspect
import os
import tempfile
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile

import pb_chime5.util.process_caller as pc
from pb_chime5.io.path_utils import normalize_path

UTILS_DIR = os.path.join(os.path.dirname(__file__), 'utils')


def load_audio(
        path,
        *,
        frames=-1,
        start=0,
        stop=None,
        dtype=np.float64,
        fill_value=None,
        expected_sample_rate=None,
        unit='samples',
        return_sample_rate=False,
):
    """
    WIP will deprecate audioread in the future

    Difference to soundfile.read:
     - Default: Return only signal
     - With the argument "unit" the unit of frames, start and stop can be
       changed (stop currently unsupported).
     - With given expected_sample_rate an assert is included (recommended)

    soundfile.read doc text and some examples:

    Provide audio data from a sound file as NumPy array.

    By default, the whole file is read from the beginning, but the
    position to start reading can be specified with `start` and the
    number of frames to read can be specified with `frames`.
    Alternatively, a range can be specified with `start` and `stop`.

    If there is less data left in the file than requested, the rest of
    the frames are filled with `fill_value`.
    If no `fill_value` is specified, a smaller array is returned.

    Parameters
    ----------
    file : str or int or file-like object
        The file to read from.  See :class:`SoundFile` for details.
    frames : int, optional
        The number of frames to read. If `frames` is negative, the whole
        rest of the file is read.  Not allowed if `stop` is given.
    start : int, optional
        Where to start reading.  A negative value counts from the end.
    stop : int, optional
        The index after the last frame to be read.  A negative value
        counts from the end.  Not allowed if `frames` is given.
    dtype : {'float64', 'float32', 'int32', 'int16'}, optional
        Data type of the returned array, by default ``'float64'``.
        Floating point audio data is typically in the range from
        ``-1.0`` to ``1.0``.  Integer data is in the range from
        ``-2**15`` to ``2**15-1`` for ``'int16'`` and from ``-2**31`` to
        ``2**31-1`` for ``'int32'``.

        .. note:: Reading int values from a float file will *not*
            scale the data to [-1.0, 1.0). If the file contains
            ``np.array([42.6], dtype='float32')``, you will read
            ``np.array([43], dtype='int32')`` for ``dtype='int32'``.

    Returns
    -------
    audiodata : numpy.ndarray or type(out)
        A two-dimensional (frames x channels) NumPy array is returned.
        If the sound file has only one channel, a one-dimensional array
        is returned.  Use ``always_2d=True`` to return a two-dimensional
        array anyway.

        If `out` was specified, it is returned.  If `out` has more
        frames than available in the file (or if `frames` is smaller
        than the length of `out`) and no `fill_value` is given, then
        only a part of `out` is overwritten and a view containing all
        valid frames is returned.

    Other Parameters
    ----------------
    always_2d : bool, optional
        By default, reading a mono sound file will return a
        one-dimensional array.  With ``always_2d=True``, audio data is
        always returned as a two-dimensional array, even if the audio
        file has only one channel.
    fill_value : float, optional
        If more frames are requested than available in the file, the
        rest of the output is be filled with `fill_value`.  If
        `fill_value` is not specified, a smaller array is returned.
    out : numpy.ndarray or subclass, optional
        If `out` is specified, the data is written into the given array
        instead of creating a new array.  In this case, the arguments
        `dtype` and `always_2d` are silently ignored!  If `frames` is
        not given, it is obtained from the length of `out`.
    samplerate, channels, format, subtype, endian, closefd
        See :class:`SoundFile`.

    Examples
    --------
    >>> from pb_chime5.io import load_audio
    >>> path = '/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav'
    >>> data = load_audio(path)
    >>> data.shape
    (46797,)

    Say you load audio examples from a very long audio, you can provide a
    start position and a duration in samples or seconds.

    >>> path = '/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav'
    >>> signal = load_audio(path, start=0, frames=16_000)
    >>> signal.shape
    (16000,)
    >>> signal = load_audio(path, start=0, frames=1, unit='seconds')
    >>> signal.shape
    (16000,)

    If the audio file is to short, only return the defined part:

    >>> signal = load_audio(path, start=0, frames=160_000)
    >>> signal.shape
    (46797,)

    >>> path = '/net/db/tidigits/tidigits/test/man/ah/111a.wav'
    >>> load_audio(path)  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    RuntimeError: /net/db/tidigits/tidigits/test/man/ah/111a.wav: NIST SPHERE file
    <BLANKLINE>

    """

    # soundfile does not support pathlib.Path.
    # ToDo: Is this sill True?
    path = normalize_path(path, as_str=True)

    if unit == 'samples':
        pass
    elif unit == 'seconds':
        if stop is not None:
            if stop < 0:
                raise NotImplementedError(unit, stop)
        with soundfile.SoundFile(path) as f:
            # total_samples = len(f)
            samplerate = f.samplerate
        start = int(np.round(start * samplerate))
        if frames > 0:
            frames = int(np.round(frames * samplerate))
        if stop is not None and stop > 0:
            stop = int(np.round(stop * samplerate))
    else:
        raise ValueError(unit)

    try:
        with soundfile.SoundFile(
                path,
                'r',
        ) as f:
            if dtype is None:
                from paderbox.utils.mapping import Dispatcher
                mapping = Dispatcher({
                    'PCM_16': np.int16,
                    'FLOAT': np.float32,
                    'DOUBLE': np.float64,
                })
                dtype = mapping[f.subtype]

            frames = f._prepare_read(start=start, stop=stop, frames=frames)
            data = f.read(frames=frames, dtype=dtype, fill_value=fill_value)
        signal, sample_rate = data, f.samplerate
    except RuntimeError as e:
        if isinstance(path, (Path, str)):
            if Path(path).suffix == '.wav':
                # Improve exception msg for NIST SPHERE files.
                from paderbox.utils.process_caller import run_process
                cp = run_process(f'file {path}')
                stdout = cp.stdout
                raise RuntimeError(f'{stdout}') from e
            else:
                raise RuntimeError(f'Wrong suffix {path.suffix} in {path}')
        raise

    if expected_sample_rate is not None:
        if expected_sample_rate != sample_rate:
            raise ValueError(
                f'Requested sampling rate is {expected_sample_rate} but the '
                f'audiofile has {sample_rate}'
            )

    # When signal is multichannel, than soundfile return (samples, channels)
    # At NT it is more common to have the shape (channels, samples)
    # => transpose
    signal = signal.T

    if return_sample_rate:
        return signal, sample_rate
    else:
        return signal


def audioread(path, offset=0.0, duration=None, expected_sample_rate=None):
    """
    Reads a wav file, converts it to 32 bit float values and reshapes according
    to the number of channels.

    This function uses the `wavefile` module which in turn uses `libsndfile` to
    read an audio file. This is much faster than the previous version based on
    `librosa`, especially if one reads a short segment of a long audio file.

    .. note:: Contrary to the previous version, this one does not implicitly
        resample the audio if the `sample_rate` parameter differs from the
        actual sampling rate of the file. Instead, it raises an error.


    :param path: Absolute or relative file path to audio file.
    :type: String.
    :param offset: Begin of loaded audio.
    :type: Scalar in seconds.
    :param duration: Duration of loaded audio.
    :type: Scalar in seconds.
    :param sample_rate: (deprecated) Former audioread did implicit resampling
        when a different sample rate was given. This raises an error if the
        `sample_rate` does not match the files sampling rate. `None` accepts
        any rate.
    :type: scalar in number of samples per second
    :return:

    .. admonition:: Example:
        Only path provided:

        >>> path = '/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav'
        >>> signal, sample_rate = audioread(path)
        >>> signal.shape
        (46797,)

        Say you load audio examples from a very long audio, you can provide a
        start position and a duration in seconds.

        >>> path = '/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav'
        >>> signal, sample_rate = audioread(path, offset=0, duration=1)
        >>> signal.shape
        (16000,)
        >>> signal, sample_rate = audioread(path, offset=0, duration=10)
        >>> signal.shape
        (160000,)

        >>> path = '/net/db/tidigits/tidigits/test/man/ah/111a.wav'
        >>> audioread(path)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        OSError: /net/db/tidigits/tidigits/test/man/ah/111a.wav: NIST SPHERE file
        <BLANKLINE>
    """
    import wavefile
    if isinstance(path, Path):
        path = str(path)
    path = os.path.expanduser(path)

    try:
        with wavefile.WaveReader(path) as wav_reader:
            channels = wav_reader.channels
            sample_rate = wav_reader.samplerate
            if expected_sample_rate is not None and expected_sample_rate != sample_rate:
                raise ValueError(
                    'Requested sampling rate is {} but the audiofile has {}'.format(
                        expected_sample_rate, sample_rate
                    )
                )

            if duration is None:
                samples = wav_reader.frames - int(np.round(offset * sample_rate))
                frames_before = int(np.round(offset * sample_rate))
            else:
                samples = int(np.round(duration * sample_rate))
                frames_before = int(np.round(offset * sample_rate))

            data = np.zeros((channels, samples), dtype=np.float32, order='F')
            wav_reader.seek(frames_before)
            wav_reader.read(data)
            return np.squeeze(data), sample_rate
    except OSError as e:
        from paderbox.utils.process_caller import run_process
        cp = run_process(f'file {path}')
        stdout = cp.stdout
        raise OSError(f'{stdout}') from e


def audio_length(path, unit='samples'):
    """

    Args:
        path:
        unit:

    Returns:

    >>> path = '/net/fastdb/chime3/audio/16kHz/isolated/dt05_caf_real/F01_050C0102_CAF.CH1.wav'
    >>> audio_length(path)
    122111
    >>> path = '/net/db/voiceHome/audio/noises/dev/home3_room2_arrayGeo3_arrayPos2_noiseCond1.wav'
    >>> audio_length(path)  # correct for multichannel
    960000
    >>> with soundfile.SoundFile(str(path)) as f:
    ...     print(f.read().shape)
    (960000, 8)
    """

    # params = soundfile.info(str(path))
    # return int(params.samplerate * params.duration)

    if unit == 'samples':
        with soundfile.SoundFile(str(path)) as f:
            return len(f)
    elif unit == 'seconds':
        with soundfile.SoundFile(str(path)) as f:
            return len(f) / f.samplerate
    else:
        return ValueError(unit)


def audio_channels(path):
    """

    >>> path = '/net/fastdb/chime3/audio/16kHz/isolated/dt05_caf_real/F01_050C0102_CAF.CH1.wav'
    >>> audio_channels(path)
    1
    >>> path = '/net/db/voiceHome/audio/noises/dev/home3_room2_arrayGeo3_arrayPos2_noiseCond1.wav'
    >>> audio_channels(path)  # correct for multichannel
    8
    """
    with soundfile.SoundFile(str(path)) as f:
        return f.channels


def audio_shape(path):
    """

    >>> path = '/net/fastdb/chime3/audio/16kHz/isolated/dt05_caf_real/F01_050C0102_CAF.CH1.wav'
    >>> audio_shape(path)
    122111
    >>> path = '/net/db/voiceHome/audio/noises/dev/home3_room2_arrayGeo3_arrayPos2_noiseCond1.wav'
    >>> audio_shape(path)  # correct for multichannel
    (8, 960000)
    >>> audioread(path)[0].shape
    (8, 960000)
    """
    with soundfile.SoundFile(str(path)) as f:
        channels = f.channels
        if channels == 1:
            return len(f)
        else:
            return channels, len(f)


def is_nist_sphere_file(path):
    """Check if given path is a nist/sphere file"""
    if not os.path.exists(path):
        return False
    cmd = f'file {path}'
    return 'NIST SPHERE file' in pc.run_process(cmd).stdout


def read_nist_wsj(path, audioread_function=audioread, **kwargs):
    """
    Converts a nist/sphere file of wsj and reads it with audioread.

    :param path: file path to audio file.
    :param audioread_function: Function to use to read the resulting audio file
    :return:
    """
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    cmd = "{}/sph2pipe -f wav {path} {dest_file}".format(
        UTILS_DIR, path=path, dest_file=tmp_file.name
    )
    pc.run_processes(cmd, ignore_return_code=False)
    signal = audioread_function(tmp_file.name, **kwargs)
    os.remove(tmp_file.name)
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
    with wave.open(str(path), 'r') as wave_file:
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
    audio_data = np.array(
        [interleaved_audio_data[ch::channels] for ch in range(channels)])
    audio_data = audio_data.astype(np.float32) / np.max(audio_data)
    return audio_data
