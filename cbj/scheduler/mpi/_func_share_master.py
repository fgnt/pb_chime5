if not __package__:
    from cbj_lib import set_package

    set_package()

from .helper import *


def share_master(iterator, disable_pbar=False, allow_single_worker=False):
    """
    A master process pushes tasks to the workers.
    Required at least 2 mpi processes, but to produce a speedup 3 are required.

    Parallel: Body of the for loop.

    Note:
        Inside the for loop a break is allowed to mark the current task as
        successful and do not calculate further tasks.


    ToDo:
        When a slave throw a exception, the task is currently ignored.
        Change it that the execution get canceled.

    """
    from tqdm import tqdm

    if allow_single_worker and size == 1:
        if disable_pbar:
            yield from iterator
        else:
            yield from tqdm(iterator)
        return

    assert size > 1, size

    status = MPI.Status()
    workers = size - 1

    comm.Barrier()

    if rank == 0:
        i = 0
        with tqdm(total=len(iterator), disable=disable_pbar) as pbar:
            pbar.set_description(f'busy: {workers}')
            while workers > 0:
                source = comm.recv(
                    source=MPI.ANY_SOURCE,
                    status=status)
                if source is not None:
                    comm.send(i, dest=source)
                    i += 1
                    pbar.update()
                else:
                    workers -= 1
                    pbar.set_description(f'busy: {workers}')

        assert workers == 0, workers
        # i is bigger than len(iterator), because the slave says value is to big
        # and than the master increases the value
        assert len(iterator) < i, f'{len(iterator)}, {i}: Iterator is not consumed'
    else:
        try:
            comm.send(rank, dest=0)
            next_index = comm.recv(source=0)
            for i, val in enumerate(iterator):
                if i == next_index:
                    yield val
                    comm.send(rank, dest=0)
                    next_index = comm.recv(source=0)
        finally:
            comm.send(None, dest=0)


if __name__ == '__main__':
    from .helper import test_relaunch_with_mpi
    test_relaunch_with_mpi()

    iter = range(5)

    comm.Barrier()

    for i in share_master(iter, disable_pbar=True):
        print('loop body:', i, rank)
        if rank == 2:
            break

    print('Exit', rank)
