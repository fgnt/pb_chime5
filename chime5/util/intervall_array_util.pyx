
# Does not work in ipynb
# from libcpp.vector cimport vector
# from libcpp.pair cimport pair


def cy_non_intersection(interval, intervals):
    cdef:
        int start
        int end
        int i_start
        int i_end
        list new_interval
    start, end = interval
    new_interval = []

    for i_start, i_end in intervals:

        if start < i_start < end:
            i_start = end
        elif start < i_end < end:
            i_end = start
        elif i_start < start and end < i_end:
            new_interval.append((i_start, start))
            i_start = end

        if i_start < i_end:
            new_interval.append((i_start, i_end))

    return tuple(new_interval)