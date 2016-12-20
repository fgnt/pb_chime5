"""

>>> from pathlib import Path
>>> p = Path('/') / 'net' # /storage/python_unittest_data
>>> p
>>> p = p / 'storage'
>>> p
>>> str(p)

"""

import os
from pathlib import Path


testing = Path(os.getenv(
    'NT_TESTING_DIR',
    '/net/storage/python_unittest_data'
))
timit = Path(os.getenv(
    'NT_TIMIT_DIR',
    '/net/speechdb/timit'
))
tidigits = Path(os.getenv(
    'NT_TIDIGITS_DIR',
    '/net/speechdb/tidigits'
))
database_jsons = Path(os.getenv(
    'NT_DATABASE_JSONS_DIR',
    '/net/storage/database_jsons'
))
chime = Path(os.getenv(
    'NT_CHIME_DIR',
    '/net/ssd/2015/chime'
))
kaldi_root = Path(os.getenv(
    'KALDI_ROOT',
    '/net/ssd/jheymann/software/kaldi_latest'
    #'/net/ssd/software/kaldi'
))
merl_mixtures = Path(os.getenv(
    'NT_MERL_MIXTURES_DIR',
    '/net/storage/db/merl_speaker_mixtures'
))

if __name__ == "__main__":
    import doctest

    doctest.testmod()
