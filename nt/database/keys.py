"""
The keys file is part of the new database concept 2017.

These are database related keys. Use them in your database JSON.

Please avoid abbreviations and discuss first, before adding new keys.
"""
SEQ_LEN = 'sequence_length'
OBS_ID = 'observation_id'  # Replaces mixture ID for multi-speaker scenario.


# Dimension prefixes for i.e. observation signal.
ARRAY = 'a'
SPEAKER = 's'
CHANNEL = 'c'


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

# Information per observation
SPEAKER_ID = 'speaker_id'
GENDER = 'gender'
START = 'start'
END = 'end'

# temporary keys, need to be discussed
EXAMPLES = "examples"
EXAMPLE_ID = "example_id"
DATASETS = "datasets"
AUDIO = "audio"
META = "meta"
TAGS = "tags"
ANNOTATION = "annotation"
PHONES = "phones"
WORDS = "words"
EVENTS = "events"
MALE = "male"
FEMALE = "female"
HIERARCHICAL_MAPPING = "hierarchical_mapping"
