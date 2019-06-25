# distutils: language = c++

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

def cy_alignment_id2phone(alignment, id2phone):
    """
    This function is around 10 times faster than a python version.
    The speedup comes from the line "cdef vector[int] v".
    """
    cdef vector[int] v
    cdef int e

    return {
        k: np.array(
            [id2phone[e] for e in v]
        )
        for k, v in alignment.items()
    }
