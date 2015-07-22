import numpy as np
import wave
import struct

def read_wav(filename, start_frame=None, number_of_frames=None):
    """
    Reads a wav file, converts it to 32 bit float values and reshapes accoring to the number of channels.
    """
    wav_file_handle = wave.open(filename)
    
    if start_frame is not None:
        wav_file_handle.setpos(start_frame)

    if number_of_frames is None:
        number_of_frames = wav_file_handle.getnframes()

    audio_raw = wav_file_handle.readframes(number_of_frames)
    
    # sample_width = wav_file_handle.getsampwidth()
    # print(sample_width)
    # print(len(audio_raw))
    # print(wav_file_handle.getparams())

    audio_data = np.array(struct.unpack('<' + wav_file_handle.getnchannels()*(number_of_frames)*"h", audio_raw),
                          dtype=np.float32)
    audio_data /= np.iinfo(np.int16).max
    audio_data = audio_data.reshape((wav_file_handle.getnchannels(), -1))
    return audio_data
