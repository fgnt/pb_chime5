"""
The keys file is part of the new database concept 2017.

These are database related keys. Use them in your database JSON.

Please avoid abbreviations and discuss first, before adding new keys.
"""
DATASETS = "datasets"
EXAMPLES = "examples"
META = "meta"

# Information per example
AUDIO_PATH = "audio_path"
AUDIO_DATA = "audio_data"
NUM_SAMPLES = 'num_samples'
NUM_FRAMES = 'num_frames'
EXAMPLE_ID = "example_id"  # Replaces mixture ID for multi-speaker scenario.
SPEAKER_ID = 'speaker_id'
GENDER = 'gender'
START = 'start'  # startingtime for speaker_id
END = 'end'  # endingtime for speaker_id

# segmentation refers to a list of tuples
# [(<label 1>, <start sample>, <end sample>), ...]
SEGMENTATION = "segmentation"
# transcription refers to a list of labels [<label 1>, <label 2>, ...] providing
# the labels that appear in an example in a certain order
TRANSCRIPTION = 'transcription'
KALDI_TRANSCRIPTION = 'kaldi_transcription'
# tags refers to a list of labels [<label 1>, <label 2>, ...] providing
# the labels that appear in an example in a any order
TAGS = "tags"

# Signals
OBSERVATION = 'observation'
OBSERVATION_LAPEL = 'observation_lapel'
OBSERVATION_HEADSET = 'observation_headset'

SPEECH_SOURCE = 'speech_source'
SPEECH_IMAGE = 'speech_image'
NOISE_IMAGE = 'noise_image'

RIR = 'rir'
RIR_DIRECT = 'rir_direct'
RIR_TAIL = 'rir_tail'

# Dimension prefixes for i.e. observation signal.
ARRAY = 'a'
SPEAKER = 's'
CHANNEL = 'c'

# other sub-keys
MALE = "male"
FEMALE = "female"
PHONES = "phones"
WORDS = "words"
EVENTS = "events"
SAMPLE_RATE = "sample_rate"

# temporary keys, need to be discussed
LABELS = "labels"
MAPPINGS = "mappings"
