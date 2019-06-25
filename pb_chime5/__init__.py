from pathlib import Path

import os
os.nice(1)  # be nice

try:
    import mkl  # assume numpy from anaconda
    mkl.set_num_threads(1)
except ModuleNotFoundError:
    pass
os.environ['OMP_NUM_THREADS'] = str(1)  # recommended for HPC systems
os.environ['GOMP_NUM_THREADS'] = str(1)  # recommended for HPC (maybe gnu omp)
os.environ['MKL_NUM_THREADS'] = str(1)
os.environ['NUMEXPR_NUM_THREADS'] = str(1)

git_root = Path(__file__).parent.parent.resolve().expanduser()

from . import (
    activity,
    mapping,
    core,
)
