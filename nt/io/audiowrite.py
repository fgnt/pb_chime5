import numpy as np
import threading
from pathlib import Path

import soundfile
from scipy.io.wavfile import write as wav_write

from nt.utils.mapping import Dispatcher

int16_max = np.iinfo(np.int16).max
int16_min = np.iinfo(np.int16).min


def dump_audio(
        obj,
        path,
        *,
        sample_rate=16000,
        dtype=np.int16,
        start=None,
        normalize=True,
):
    """
    If normalize is False and the dytpe is float, the values of obj should be in
    the range [-1, 1).

    >>> from nt.utils.process_caller import run_process
    >>> from nt.io import load_audio
    >>> a = np.array([1, 2, -4, 4], dtype=np.int16)
    >>> import io, os
    >>> # file = io.BytesIO()
    >>> file = Path('tmp_audio_data.wav')
    >>> dump_audio(a, file, normalize=False)
    >>> load_audio(file) * 2**15
    array([ 1.,  2., -4.,  4.])
    >>> print(run_process(f'file {file}').stdout)
    tmp_audio_data.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
    <BLANKLINE>
    >>> dump_audio(a, file, normalize=True)
    >>> load_audio(file)
    array([ 0.24996948,  0.49996948, -0.99996948,  0.99996948])
    >>> print(run_process(f'file {file}').stdout)
    tmp_audio_data.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
    <BLANKLINE>

    >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) / 32
    >>> data
    array([0.     , 0.03125, 0.0625 , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.28125])
    >>> dump_audio(data, file, normalize=False)
    >>> load_audio(file)
    array([0.     , 0.03125, 0.0625 , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.28125])
    >>> print(run_process(f'file {file}').stdout)
    tmp_audio_data.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
    <BLANKLINE>
    >>> dump_audio(np.array([16, 24]) / 32, file, normalize=False, start=1)
    >>> load_audio(file)
    array([0.     , 0.5    , 0.75   , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.28125])
    >>> print(run_process(f'file {file}').stdout)
    tmp_audio_data.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
    <BLANKLINE>
    >>> dump_audio(np.array([16, 24, 24, 24]) / 32, file, normalize=False, start=9)
    >>> load_audio(file)
    array([0.     , 0.5    , 0.75   , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.5    , 0.75   , 0.75   , 0.75   ])
    >>> load_audio(file).shape
    (13,)
    >>> dump_audio(np.array([16, 24, 24, 24]) / 32, file, normalize=False, start=20)
    >>> load_audio(file)
    array([0.     , 0.5    , 0.75   , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.5    , 0.75   , 0.75   , 0.75   , 0.     ,
           0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.5    ,
           0.75   , 0.75   , 0.75   ])
    >>> load_audio(file).shape
    (24,)
    >>> print(run_process(f'file {file}').stdout)
    tmp_audio_data.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
    <BLANKLINE>
    >>> os.remove('tmp_audio_data.wav')
    >>> dump_audio(np.array([16, 24, 24, 24]) / 32, file, normalize=False, start=20)
    >>> load_audio(file)
    array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
           0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.75,
           0.75, 0.75])
    >>> load_audio(file).shape
    (24,)
    >>> print(run_process(f'file {file}').stdout)
    tmp_audio_data.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
    <BLANKLINE>

    >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) / 32
    >>> data
    array([0.     , 0.03125, 0.0625 , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.28125])
    >>> dump_audio(data, file, normalize=False, dtype=None)
    >>> load_audio(file)
    array([0.     , 0.03125, 0.0625 , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.28125])
    >>> print(run_process(f'soxi {file}').stdout)
    <BLANKLINE>
    Input File     : 'tmp_audio_data.wav'
    Channels       : 1
    Sample Rate    : 16000
    Precision      : 53-bit
    Duration       : 00:00:00.00 = 10 samples ~ 0.046875 CDDA sectors
    File Size      : 160
    Bit Rate       : 2.05M
    Sample Encoding: 64-bit Floating Point PCM
    <BLANKLINE>
    <BLANKLINE>
    >>> dump_audio(data.astype(np.float32), file, normalize=False, dtype=None)
    >>> load_audio(file, dtype=None)
    array([0.     , 0.03125, 0.0625 , 0.09375, 0.125  , 0.15625, 0.1875 ,
           0.21875, 0.25   , 0.28125], dtype=float32)
    >>> print(run_process(f'soxi {file}').stdout)
    <BLANKLINE>
    Input File     : 'tmp_audio_data.wav'
    Channels       : 1
    Sample Rate    : 16000
    Precision      : 24-bit
    Duration       : 00:00:00.00 = 10 samples ~ 0.046875 CDDA sectors
    File Size      : 120
    Bit Rate       : 1.54M
    Sample Encoding: 32-bit Floating Point PCM
    <BLANKLINE>
    <BLANKLINE>

    """
    if isinstance(path, Path):
        path = str(path)

    if normalize:
        if not obj.dtype.kind in ['f', 'i']:
            raise TypeError(
                'Only float and int is currently supported with normalize. '
                f'Got dtype {obj.dtype}'
            )
        # Normalization can change the type (e.g. int to float).
        # When saving as float, normalize is a bad idea.
        # The normalization is adjusted for int16
        assert dtype == np.int16, (
            'Currently is only normalize allowed for dtype == np.int16'
            f'and not for dtype == {dtype}'
        )
        # Correction, because the allowed values are in the range [-1, 1).
        # => "1" is not a vaild value
        correction = (2**15 - 1) / (2**15)
        obj = obj * correction / np.amax(np.abs(obj))

    if start is None or not Path(path).exists():
        obj = np.asarray(obj)
        if obj.ndim == 1:
            channels = 1
        else:
            channels = obj.shape[0]

        sf_args = dict(
            mode='w',
            channels=channels,
            samplerate=sample_rate,
        )
    else:
        sf_args = dict(
            mode='r+'
        )

    dtype_map = Dispatcher({
        np.float32: 'FLOAT',
        np.dtype('float32'): 'FLOAT',
        np.float64: 'DOUBLE',
        np.dtype('float64'): 'DOUBLE',
    })

    if dtype in [np.int16]:
        pass
    elif dtype in [np.float32, np.float64]:
        sf_args['subtype'] = dtype_map[dtype]
    elif dtype is None:
        sf_args['subtype'] = dtype_map[obj.dtype]
    else:
        raise TypeError(dtype)

    # soundfile.write()

    with soundfile.SoundFile(path, **sf_args) as f:
        if start is not None:
            f.seek(start)
        f.write(obj.T)
    return


def audiowrite(data, path, sample_rate=16000, normalize=False, threaded=True):
    """ Write the audio data ``data`` to the wav file ``path``

    The file can be written in a threaded mode. In this case, the writing
    process will be started at a separate thread. Consequently, the file will
    not be written when this function exits.

    :param data: A numpy array with the audio data
    :param path: The wav file the data should be written to
    :param sample_rate: Sample rate of the audio data
    :param normalize: Normalize the audio first so that the values are within
        the range of [INTMIN, INTMAX]. E.g. no clipping occurs
    :param threaded: If true, the write process will be started as a separate
        thread
    :return: The number of clipped samples
    """
    assert data.dtype.kind in ['i', 'f'], (data.shape, data.dtype)

    if isinstance(path, Path):
        path = str(path)

    data = data.copy()

    if normalize:
        if not data.dtype.kind == 'f':
            data = data.astype(np.float)
        data /= np.maximum(np.amax(np.abs(data)), 1e-6)

    if data.dtype.kind == 'f':
        data *= int16_max

    sample_to_clip = np.sum(data > int16_max)
    if sample_to_clip > 0:
        print('Warning, clipping {} sample{}.'.format(
            sample_to_clip, '' if sample_to_clip == 1 else 's'
            ))
    data = np.clip(data, int16_min, int16_max)
    data = data.astype(np.int16)

    if threaded:
        threading.Thread(target=wav_write, args=(path, sample_rate, data)
                         ).start()
    else:
        try:
            wav_write(path, sample_rate, data)
        except Exception:  # _struct.error
            if data.ndim == 2:
                assert data.shape[1] < 20, (
                    f"channels bigger than 20 looks wrong "
                    f"(shape: {data.shape}). "
                    f"Maybe you must call audiowrite(data.T, ...)"
                )
            raise

    return sample_to_clip
