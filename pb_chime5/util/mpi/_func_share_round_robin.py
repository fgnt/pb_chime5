if not __package__:
    from cbj_lib import set_package

    set_package()

from . import core as mpi


def share_round_robin(iterator, disable_pbar=True):
    """
    Shares the work in round robin fashion

    >>> assert mpi.SIZE==2  # mpi size is 2
    >>> if mpi.RANK==0:
    ...     print(list(share_round_robin(range(5))))
    [0, 2, 4]
    >>> if mpi.RANK==1:
    ...     print(list(share_round_robin(range(2))))
    [1, 3]
    """
    from itertools import islice
    if disable_pbar or not mpi.IS_MASTER:
        return islice(iterator, mpi.RANK, None, mpi.SIZE)
    else:
        from tqdm import tqdm

        def gen():
            with tqdm(total=len(iterator), desc='MasterPbar') as pbar:
                for ele in islice(iterator, mpi.RANK, None, mpi.SIZE):
                    yield ele
                    pbar.update(mpi.SIZE)

        return gen()
        # return tqdm(islice(iterator, rank, None, size), desc='MasterPbar')


if __name__ == '__main__':
    # from .helper import test_relaunch_with_mpi
    # test_relaunch_with_mpi()

    it = range(6)

    mpi.COMM.Barrier()

    for i in share_round_robin(it):
        print('loop body:', i, mpi.RANK)

    # it = range(60)
    # for i in share_round_robin(it, disable_pbar=False):
        # import time
        # time.sleep(0.1)

    print('Exit', mpi.RANK)
