
# Does not work in ipynb
# from libcpp.vector cimport vector
# from libcpp.pair cimport pair

# ToDO: better place for testcode
#       http://ntsvr1:1619/notebooks/chime5/2018_05_17_tf_blstm.ipynb

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


def cy_intersection(interval, intervals):
    cdef:
        int start
        int end
        int i_start
        int i_end
        list new_interval

    start, end = interval
    new_interval = []

    for i_start, i_end in intervals:
        i_start = max(start, i_start)
        i_end = min(end, i_end)
        if i_start < i_end:
            new_interval.append((i_start, i_end))

    return tuple(new_interval)


def cy_parse_item(item, shape):

    cdef:
        int start
        int end
        int v
        int size

    size = shape[-1]

    assert isinstance(item, (slice)), (type(item), item)
    assert item.step is None, (item)

    if item.start is None:
        start = 0
    else:
        start = item.start
    if item.stop is None:
        stop = size
    else:
        stop = item.stop

    assert start >= 0, (start, item)
    assert start <= size, (start, item)
    assert stop >= 0, (stop, item)
    assert stop <= size, (stop, item)

    if start < 0:
        start = start % size
    if stop < 0:
        stop = start % size

    return start, stop


def cy_str_to_intervalls(string):

    cdef:
        str intervalls_string
        str intervall_string
        str start_str
        str end_str
        int start
        int end
        list intervals

    intervalls_string = string

#     start, end = interval
    intervals = []

#     for i_start, i_end in intervals:

    for intervall_string in intervalls_string.replace(' ', '').strip(',').split(','):

        try:
            start_str, end_str = intervall_string.split(':')
        except Exception as e:
            raise Exception(intervall_string) from e
        start = int(start_str)
        end = int(end_str)

        intervals.append((start, end))

    return tuple(intervals)