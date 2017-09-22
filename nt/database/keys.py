"""
These are database related keys. Use them in your database JSON.

Please avoid abbreviations and discuss first, before adding new keys.
"""
SEQ_LEN = 'sequence_length'
UTT_ID = 'utterance_id'  # Replaces mixture ID for multi-speaker scenario.


# Signals
SPEECH_SOURCE = 'speech_source'

SPEECH_IMAGE = 'speech_image'
NOISE_IMAGE = 'noise_image'

OBSERVATION = 'observation'
OBSERVATION_LAPEL = 'observation_lapel'
OBSERVATION_HEADSET = 'observation_headset'

RIR = 'rir'
RIR_DIRECT = 'rir_direct'
RIR_TAIL = 'rir_tail'


# Transcription as plain string
TRANSCRIPTION = 'transcription'
KALDI_TRANSCRIPTION = 'kaldi_transcription'
