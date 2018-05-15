"""Wraps imports for mpi4py to allow code to run on non MPI machines, too."""

_mpi_available = True

try:
    from mpi4py import MPI
    _mpi_available = False
except ImportError:
    import os
    if 'CCS' in os.environ:
        # CCS indicate PC2
        raise

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


def map_unordered(func, iterator, progress_bar=False):
    """
    A master process push tasks to the workers and receives the result.
    Required at least 2 mpi processes, but to produce a speedup 3 are required.
    Only rank 0 get the results.
    This map is lazy.

    Assume function body is fast.

    Parallel: The execution of func.

    """
    from tqdm import tqdm
    import itertools
    from enum import IntEnum, auto

    if SIZE == 1:
        if progress_bar:
            yield from tqdm(map(func, iterator))
            return
        else:
            yield from map(func, iterator)
            return

    status = MPI.Status()
    workers = SIZE - 1

    class tags(IntEnum):
        """Avoids magic constants."""
        start = auto()
        stop = auto()
        default = auto()

    COMM.Barrier()

    if RANK == 0:
        i = 0
        with tqdm(total=len(iterator), disable=not progress_bar) as pbar:
            pbar.set_description(f'busy: {workers}')
            print('asdf')
            while workers > 0:
                result = COMM.recv(
                    source=MPI.ANY_SOURCE,
                    tag=MPI.ANY_TAG,
                    status=status)
                if status.tag == tags.default:
                    COMM.send(i, dest=status.source)
                    yield result
                    i += 1
                    pbar.update()
                elif status.tag == tags.start:
                    COMM.send(i, dest=status.source)
                    i += 1
                    pbar.update()
                elif status.tag == tags.stop:
                    workers -= 1
                    pbar.set_description(f'busy: {workers}')
                else:
                    raise ValueError(status.tag)

        assert workers == 0
    else:
        try:
            COMM.send(None, dest=0, tag=tags.start)
            next_index = COMM.recv(source=0)
            for i, val in enumerate(iterator):
                if i == next_index:
                    result = func(val)
                    COMM.send(result, dest=0, tag=tags.default)
                    next_index = COMM.recv(source=0)
        finally:
            COMM.send(None, dest=0, tag=tags.stop)
