if not __package__:
    from cbj_lib import set_package

    set_package()

from .helper import *


def share_round_robin(iterator):
    """
    Shares the work in round robin fashion

    >>> assert size==2  # mpi size is 2
    >>> if rank==0:
    ...     print(list(share_round_robin(range(5))))
    [0, 2, 4]
    >>> if rank==1:
    ...     print(list(share_round_robin(range(2))))
    [1, 3]
    """
    from itertools import islice
    return islice(iterator, rank, None, size)


if __name__ == '__main__':
    from .helper import test_relaunch_with_mpi
    test_relaunch_with_mpi()

    it = range(6)

    comm.Barrier()

    for i in share_round_robin(it):
        print('loop body:', i, rank)

    print('Exit', rank)
