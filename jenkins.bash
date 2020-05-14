#!/usr/bin/env bash

# internal script for jenkins
set -e
renice -n 20 $$
source /net/software/python/2018_12/anaconda/bin/activate

# set a prefix for each cmd
green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

# Force Exit 0
# trap 'exit 0' EXIT SIGINT SIGTERM

# Use a pseudo virtualenv, http://stackoverflow.com/questions/2915471/install-a-python-package-into-a-different-directory-using-pip
mkdir -p venv
export PYTHONUSERBASE=$(readlink -m venv)

git clone git@ntgit.upb.de:python/toolbox internal_toolbox
source internal_toolbox/bash/cuda.bash
source internal_toolbox/bash/kaldi.bash

# git clone https://github.com/fgnt/pb_chime5
# cd pb_chime5

git submodule init
git submodule update
pip install --user -e pb_bss/
pip install --user -e .


make cache/chime5.json
make cache/annotation/S02.pkl
python -m pb_chime5.scripts.run test_run with session_id=dev
python -m pb_chime5.scripts.run test_run with session_id=dev wpe=False activity_type=path activity_path=cache/word_non_sil_alignment

mkdir kaldi_run_storage_dir
python -m pb_chime5.scripts.kaldi_run with storage_dir=kaldi_run_storage_dir session_id=dev job_id=1 number_of_jobs=2000

make cache/CHiME6
make cache/chime6.json
python -m pb_chime5.scripts.run test_run with session_id=dev database_path=cache/chime6.json chime6=True
mkdir kaldi_run_storage_dir_chime6
python -m pb_chime5.scripts.kaldi_run with storage_dir=kaldi_run_storage_dir_chime6 session_id=dev job_id=1 number_of_jobs=2000 database_path=cache/chime6.json chime6=True

mkdir kaldi_run_storage_dir_chime6_rttm
python -m pb_chime5.scripts.kaldi_run_rttm with \
  storage_dir=kaldi_run_storage_dir_chime6 \
  database_rttm="https://raw.githubusercontent.com/nateanl/chime6_rttm/master/dev_rttm" \
  session_id=dev \
  job_id=1 \
  number_of_jobs=6000 \
  context_samples=16000 \
  multiarray='first_array_mics'

ls
echo `pwd`
