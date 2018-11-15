# internal script for jenkins

renice -n 20 $$
source /net/ssd/software/conda/bin/activate

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

# git clone https://github.com/fgnt/pb_chime5
# cd pb_chime5

git submodule init
git submodule update
pip install --user -e pb_bss/
pip install --user -e toolbox/
pip install --user -e .
make CHiME5
make cache/chime5_orig.json
python -m pb_chime5.scripts.run test_run with session_id=dev

ls
echo `pwd`
