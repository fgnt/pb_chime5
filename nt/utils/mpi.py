"""Wraps imports for mpi4py to allow code to run on non MPI machines, too."""

_mpi_available = True

try:
    from mpi4py import MPI
    _mpi_available = False
except ImportError:
    class DUMMY_COMM_WORLD:
        size = 1
        rank = 0
        Barrier = lambda self: None
        bcast = lambda self, data, *args, **kwargs: data
        gather = lambda self, data, *args, **kwargs: [data]

    class _dummy_MPI:
        COMM_WORLD = DUMMY_COMM_WORLD()

    MPI = _dummy_MPI()

class RankInt(int):
    def __bool__(self):
        raise NotImplementedError(
            'Bool is disabled for rank. '
            'It is likly that you want to use IS_MASTER.'
        )

COMM = MPI.COMM_WORLD
RANK = RankInt(COMM.rank)
SIZE = COMM.size
MASTER = RankInt(0)
IS_MASTER = (RANK == MASTER)
