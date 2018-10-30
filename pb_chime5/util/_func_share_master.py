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
    # registered_workers = set()

    # print(f'{rank} reached Barrier in share_master')
    comm.Barrier()
    # print(f'{rank} left Barrier in share_master')


    from enum import IntEnum, auto

    class tags(IntEnum):
        start = auto()
        stop = auto()
        default = auto()
        failed = auto()

    failed_indices = []

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
                last_index = comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=MPI.ANY_TAG,
                    status=status,
                )

                if status.tag in [tags.default, tags.start]:
                    comm.send(i, dest=status.source)
                    i += 1

                if status.tag in [tags.default, tags.failed]:
                    pbar.update()

                if status.tag in [tags.stop, tags.failed]:
                    workers -= 1
                    pbar.set_description(f'{pbar_prefix}busy: {workers}')

                if status.tag == tags.failed:
                    failed_indices += [(status.source, last_index)]

        assert workers == 0, workers
        # i is bigger than len(iterator), because the slave says value is to big
        # and than the master increases the value
        if length is not None:
            if (not length < i) or len(failed_indices) > 0:
                failed_indices = '\n'.join([
                    f'worker {rank_} failed for index {index}'
                    for rank_, index in failed_indices
                ])
                raise AssertionError(
                    f'{length}, {i}: Iterator is not consumed.\n'
                    f'{failed_indices}'
                )
            # assert length < i, f'{length}, {i}: Iterator is not consumed'
    else:
        next_index = -1
        successfull = False
        try:
            # comm.send(rank, dest=0)
            comm.send(None, dest=0, tag=tags.start)
            next_index = comm.recv(source=0)
            for i, val in enumerate(iterator):
                if i == next_index:
                    yield val
                    comm.send(next_index, dest=0, tag=tags.default)
                    next_index = comm.recv(source=0)
            successfull = True
        finally:
            if successfull:
                comm.send(next_index, dest=0, tag=tags.stop)
            else:
                comm.send(next_index, dest=0, tag=tags.failed)


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
    #
    # for i in share_master(iter, disable_pbar=False):
    #     print('2 loop body:', i, rank)
    #     # if rank == 1:
    #     #     break
    #
    # for i in share_master(iter, disable_pbar=False):
    #     time.sleep(2)

    for i in share_master(iter, disable_pbar=False):
        # if i % 2:
        #     break
        pass
        time.sleep(3)
    time.sleep(2)
    print('Exit', rank)
