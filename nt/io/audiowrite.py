import numpy
from scipy.io.wavfile import write as wav_write
import threading

int16_max = numpy.iinfo(numpy.int16).max
int16_min = numpy.iinfo(numpy.int16).min

def audiowrite(data, path, samplerate=16000, normalize=False, threaded=True):

    data = data.copy()

    if normalize:
        if not data.dtype.kind == 'f':
            data = data.astype(numpy.float)
        data /= numpy.max(numpy.abs(data))

    if data.dtype.kind == 'f':
        data *= int16_max

    sample_to_clip = numpy.sum(data > int16_max)
    if sample_to_clip > 0:
        print('Warning, clipping {} samples'.format(sample_to_clip))
    data = numpy.clip(data, int16_min, int16_max)
    data = data.astype(numpy.int16)

    if threaded:
        threading.Thread(target=wav_write, args=(path, samplerate, data)).start()
    else:
        wav_write(path, samplerate, data)

    return sample_to_clip

