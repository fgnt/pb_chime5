if not __package__:
    from cbj_lib import set_package

    set_package()

from .helper import *


def share_master(
        iterator,
        # length=None,
        disable_pbar=False,
        allow_single_worker=False,
        pbar_prefix=None,
):
    """
    A master process pushes tasks to the workers.
    Required at least 2 mpi processes, but to produce a speedup 3 are required.

    Parallel: Body of the for loop.
    Communication: Indices
    Redundant computation: Each process (except the master) consumes the
                           iterator.

    Note:
        Inside the for loop a break is allowed to mark the current task as
        successful and do not calculate further tasks.


    ToDo:
        - Make a more efficient scheduling
          - allow indexable
          - Use round robin in the beginning and for the last task this
            scheduler. Or submit chunks of work.
        - When a slave throw a exception, the task is currently ignored.
          Change it that the execution get canceled.

    """
    from tqdm import tqdm

    if allow_single_worker and size == 1:
        if disable_pbar:
            yield from iterator
        else:
            yield from tqdm(iterator, mininterval=2)
        return

    assert size > 1, size

    status = MPI.Status()
    workers = size - 1

    # print(f'{rank} reached Barrier in share_master')
    comm.Barrier()
    # print(f'{rank} left Barrier in share_master')

    if rank == 0:
        i = 0
        try:
            length = len(iterator)
        except Exception:
            length = None

        if pbar_prefix is None:
            pbar_prefix = ''
        else:
            pbar_prefix = f'{pbar_prefix}, '

        with tqdm(
                total=length, disable=disable_pbar, mininterval=2, smoothing=None
        ) as pbar:
            pbar.set_description(f'{pbar_prefix}busy: {workers}')
            while workers > 0:
                source = comm.recv(
                    source=MPI.ANY_SOURCE,
                    status=status)
                if source is not None:
                    comm.send(i, dest=source)
                    i += 1
                    if length is not None:
                        if length >= i:
                            pbar.update()
                    else:
                        pbar.update()
                else:
                    workers -= 1
                    pbar.set_description(f'{pbar_prefix}busy: {workers}')

        assert workers == 0, workers
        # i is bigger than len(iterator), because the slave says value is to big
        # and than the master increases the value
        if length is not None:
            assert length < i, f'{length}, {i}: Iterator is not consumed'
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

    import tqdm
    import time
    tqdm.tqdm.monitor_interval = 0

    iter = range(5)

    comm.Barrier()

    for i in share_master(iter, disable_pbar=False):
        print('loop body:', i, rank)
        if rank == 2:
            time.sleep(0.2)
            # break

    for i in share_master(iter, disable_pbar=False):
        print('2 loop body:', i, rank)
        # if rank == 1:
        #     break

    print('Exit', rank)
