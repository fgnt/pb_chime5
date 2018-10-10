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


def _get_path(environment_name, default):
    return Path(os.getenv(environment_name, default)).expanduser()


database_jsons = _get_path(
    'NT_DATABASE_JSONS_DIR',
    '/net/vol/jenkins/jsons'
)
db_dir = _get_path(
    'NT_DB_DIR',
    '/net/db'
)
fast_db_dir = _get_path(
    'NT_FAST_DB_DIR',
    '/net/fastdb'
)
testing = _get_path(
    'NT_TESTING_DIR',
    '/net/storage/python_unittest_data'
)
kaldi_root = _get_path(
    'KALDI_ROOT',
    '/net/vol/jenkins/kaldi/2018-01-10_15-43-29_a0b71317df1035bd3c6fa49a2b6bb33c801b56ac')
matlab_toolbox = _get_path(
    'MATLAB_TOOLBOX_DIR',
    '/net/ssd/software/matlab_toolbox'
)
matlab_r2015a = _get_path(
    'MATLAB_R2015a',
    '/net/ssd/software/MATLAB/R2015a'
)
matlab_license = _get_path(
    'MATLAB_LICENSE',
    '/opt/MATLAB/R2016b_studis/licenses/network.lic'
)

ami = _get_path(
    'NT_AMI_DIR',
    db_dir / 'ami'
)
audioset = _get_path(
    'NT_AUDIOSET_DIR',
    fast_db_dir / 'AudioSet'
)
dcase_2017_task_3 = _get_path(
    'NT_DCASE_2017_TASK_3_DIR',
    fast_db_dir / 'DCASE2017' / 'Task3'
)
dcase_2017_task_4 = _get_path(
    'NT_DCASE_2017_TASK_4_DIR',
    fast_db_dir / 'DCASE2017' / 'Task4'
)
timit = _get_path(
    'NT_TIMIT_DIR',
    db_dir / 'timit'
)
tidigits = _get_path(
    'NT_TIDIGITS_DIR',
    db_dir / 'tidigits'
)
chime_3 = _get_path(
    'NT_CHIME_3_DIR',
    fast_db_dir / 'chime3'
)
chime_4 = _get_path(
    'NT_CHIME_4_DIR',
    fast_db_dir / 'chime4'
)
chime_5 = _get_path(
    'NT_CHIME_5_DIR',
    fast_db_dir / 'chime5'
)
merl_mixtures = _get_path(
    'NT_MERL_MIXTURES_DIR',
    '/net/db/merl_speaker_mixtures'
)
merl_mixtures_mc = _get_path(
    'NT_MERL_MIXTURES_MC_DIR',
    "/net/vol/ldrude/projects/2017/project_dc_storage/merl_mixtures_mc_v6/"
)
german_speechdata = _get_path(
    'NT_GERMAN_SPEECHDATA_DIR',
    '/net/storage/jheymann/speech_db/german-speechdata-package-v2/'
)
noisex92 = _get_path(
    'NT_NoiseX_92_DIR',
    db_dir / 'NoiseX_92'
)
reverb = _get_path(
    'NT_REVERB_DIR',
    fast_db_dir / 'reverb'
)
wsj = _get_path(
    'NT_WSJ_DIR',
    fast_db_dir / 'wsj'
)
wsj_corrected_paths = _get_path(
    'NT_WSJ_DIR',
    db_dir / 'wsj_corrected_paths'
)
wsjcam0 = _get_path(
    'NT_WSJCAM0_DIR',
    db_dir / 'wsjcam0'
)
wsj_bss = _get_path(
    'NT_WSJ_BSS_DIR',
    fast_db_dir / 'wsj_bss'
)
language_model = _get_path(
    'LANGUAGE_MODEL',
    '/net/storage/jheymann/__share/ldrude/2016/2016-05-10_lm'
)
wsj_mc = _get_path(
    'NT_WSJ_MC_DIR',
    db_dir / 'wsj_mc_complete'
)
librispeech = _get_path(
    'NT_LIBRISPEECH_DIR',
    db_dir / 'LibriSpeech'
)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
