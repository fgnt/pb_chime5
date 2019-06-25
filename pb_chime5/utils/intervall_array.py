import numpy as np
from pb_chime5.utils.intervall_array_util import (
    cy_non_intersection,
    cy_intersection,
    cy_parse_item,
    cy_str_to_intervalls,
)


def ArrayIntervall_from_str(string, shape):
    """
    >>> ArrayIntervall_from_str('1:4, 5:20, 21:25', shape=50)
    ArrayIntervall("1:4, 5:20, 21:25", shape=(50,))
    >>> ArrayIntervall_from_str('1:4', shape=50)
    ArrayIntervall("1:4", shape=(50,))
    >>> ArrayIntervall_from_str('1:4,', shape=50)
    ArrayIntervall("1:4", shape=(50,))
    >>> ArrayIntervall_from_str('0:142464640,', shape=242464640)
    ArrayIntervall("0:142464640", shape=(50,))

    """
    ai = ArrayIntervall(shape)
    if string == '':
        print('empty intervall found')
        pass
    else:
        if not ',' in string:
            string = string + ','
        try:
            ai.add_intervals_from_str(string)
            # intervalls = eval(f'np.s_[{string}]')
            # ai.add_intervals(intervalls)
            # for item in eval(f'np.s_[{string}]'):
            #     try:
            #         ai[item] = 1
            #     except Exception as e:
            #         raise Exception(item, ai) from e
        except Exception as e:
            raise Exception(string) from e
    return ai


class ArrayIntervall:
    from_str = staticmethod(ArrayIntervall_from_str)

    @staticmethod
    def from_array(array):
        """
        >>> ai = ArrayIntervall.from_array(np.array([1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=np.bool))
        >>> ai
        ArrayIntervall("0:2, 3:4, 6:8", shape=(9,))
        >>> ai[:]
        array([ True,  True, False,  True, False, False,  True,  True, False])
        >>> a = np.array([1, 1, 1, 1], dtype=np.bool)
        >>> assert all(a == ArrayIntervall.from_array(a)[:])
        >>> a = np.array([0, 0, 0, 0], dtype=np.bool)
        >>> assert all(a == ArrayIntervall.from_array(a)[:])
        >>> a = np.array([0, 1, 1, 0], dtype=np.bool)
        >>> assert all(a == ArrayIntervall.from_array(a)[:])
        >>> a = np.array([1, 0, 0, 1], dtype=np.bool)
        >>> assert all(a == ArrayIntervall.from_array(a)[:])

        """
        array = np.asarray(array)
        assert array.ndim == 1, (array.ndim, array)
        assert array.dtype == np.bool, (np.bool, array)

        diff = np.diff(array.astype(np.int32))

        rising = list(np.atleast_1d(np.squeeze(np.where(diff > 0), axis=0)) + 1)
        falling = list(np.atleast_1d(np.squeeze(np.where(diff < 0), axis=0)) + 1)

        if array[0] == 1:
            rising = [0] + rising

        if array[-1] == 1:
            falling = falling + [len(array)]

        ai = ArrayIntervall(shape=array.shape)
        for start, stop in zip(rising, falling):
            ai[start:stop] = 1

        return ai

    def __reduce__(self):
        """
        >>> from IPython.lib.pretty import pprint
        >>> import pickle
        >>> import jsonpickle, json
        >>> ai = ArrayIntervall_from_str('1:4, 5:20, 21:25', shape=50)
        >>> ai
        ArrayIntervall("1:4, 5:20, 21:25", shape=(50,))
        >>> pickle.loads(pickle.dumps(ai))
        ArrayIntervall("1:4, 5:20, 21:25", shape=(50,))
        >>> jsonpickle.loads(jsonpickle.dumps(ai))
        ArrayIntervall("1:4, 5:20, 21:25", shape=(50,))
        >>> pprint(json.loads(jsonpickle.dumps(ai)))
        {'py/reduce': [{'py/function': 'chime5.util.intervall_array.ArrayIntervall_from_str'},
          {'py/tuple': ['1:4, 5:20, 21:25', 50]},
          None,
          None,
          None]}
        """
        return self.from_str, (self._intervals_as_str, self.shape[-1])

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = [shape]

        assert len(shape) == 1, shape

        self.shape = tuple(shape)
        # self._intervals = (,)

    _intervals_normalized = True
    # _normalized_intervals = ()
    _intervals = ()

    def __len__(self):
        return self.shape[0]

    @property
    def normalized_intervals(self):
        if not self._intervals_normalized:
            # print('normalized_intervals', self._intervals)
            self._intervals = self._normalize(self._intervals)
            # print('normalized_intervals', self._intervals)
        return self._intervals

    @property
    def intervals(self):
        return self._intervals

    @intervals.setter
    def intervals(self, item):
        self._intervals_normalized = False
        self._intervals = tuple(item)

    # @intervals.setter
    # def intervals(self, value):
    #     self._intervals_normalized = False
    #     self._intervals = value

    @staticmethod
    def _normalize(intervals):
        """
        >>> ArrayIntervall._normalize([])
        ()
        >>> ArrayIntervall._normalize([(0, 1)])
        ((0, 1),)
        >>> ArrayIntervall._normalize([(0, 1), (2, 3)])
        ((0, 1), (2, 3))
        >>> ArrayIntervall._normalize([(0, 1), (20, 30)])
        ((0, 1), (20, 30))
        >>> ArrayIntervall._normalize([(0, 1), (1, 3)])
        ((0, 3),)
        >>> ArrayIntervall._normalize([(0, 1), (1, 3), (3, 10)])
        ((0, 10),)
        """
        intervals = [(s, e) for s, e in sorted(intervals) if s < e]
        for i in range(len(intervals)):
            try:
                s, e = intervals[i]
            except IndexError:
                break
            try:
                next_s, next_e = intervals[i+1]
                while next_s <= e:
                    e = max(e, next_e)
                    del intervals[i+1]
                    next_s, next_e = intervals[i+1]
            except IndexError:
                pass
            finally:
                intervals[i] = (s, e)
        return tuple(intervals)

    @property
    def _intervals_as_str(self):
        i_str = []
        for i in self.normalized_intervals:
            start, end = i
            #             i_str += [f'[{start}, {end})']
            i_str += [f'{start}:{end}']

        i_str = ', '.join(i_str)
        return i_str

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._intervals_as_str}", shape={self.shape})'

    def _parse_item(self, item):
        assert isinstance(item, (slice)), (type(item), item)
        assert item.step is None, (item)

        start = item.start
        stop = item.stop

        if start is None:
            start = 0
        if stop is None:
            stop = self.shape[-1]

        for v in [start, stop]:
            assert v >= 0, (v, item)
            assert v <= self.shape[-1], (v, item)

        size = self.shape[-1]

        if start < 0:
            start = start % size
        if stop < 0:
            stop = start % size

        return start, stop

    def add_intervals_from_str(self, string_intervals):

        self.intervals = self.intervals + cy_str_to_intervalls(string_intervals)

    def add_intervals(self, intervals):
        """
        for item in intervals:
            self[item] = 1

        # This function is equal to above example code, but significant faster.
        """
        # Short circuit
        self.intervals = self.intervals + tuple(
            [cy_parse_item(i, self.shape) for i in intervals]
        )

    def __setitem__(self, item, value):
        """

        >>> ai = ArrayIntervall(50)
        >>> ai[10:15] = 1
        >>> ai
        ArrayIntervall("10:15", shape=(50,))
        >>> ai[5:10] = 1
        >>> ai
        ArrayIntervall("5:15", shape=(50,))
        >>> ai[1:4] = 1
        >>> ai
        ArrayIntervall("1:4, 5:15", shape=(50,))
        >>> ai[15:20] = 1
        >>> ai
        ArrayIntervall("1:4, 5:20", shape=(50,))
        >>> ai[21:25] = 1
        >>> ai
        ArrayIntervall("1:4, 5:20, 21:25", shape=(50,))
        >>> ai[10:15] = 1
        >>> ai
        ArrayIntervall("1:4, 5:20, 21:25", shape=(50,))
        >>> ai[0:50] = 1
        >>> ai[0:0] = 1
        >>> ai
        ArrayIntervall("0:50", shape=(50,))
        >>> ai[3:6]
        array([ True,  True,  True])
        >>> ai[3:6] = np.array([ True,  False,  True])
        >>> ai
        ArrayIntervall("0:4, 5:50", shape=(50,))
        >>> ai[10:13] = np.array([ False,  True,  False])
        >>> ai
        ArrayIntervall("0:4, 5:10, 11:12, 13:50", shape=(50,))

        """

        start, stop = cy_parse_item(item, self.shape)

        if np.isscalar(value) and value == 1:
            self.intervals = self.intervals + ((start, stop),)
            # self.intervals = self._union([start, stop], self.intervals)
        elif isinstance(value, (tuple, list, np.ndarray)):
            assert len(value) == stop - start, (start, stop, len(value), value)
            ai = ArrayIntervall.from_array(value)
            intervals = self.intervals
            # intervals = self._non_intersection([start, stop], intervals)

            intervals = cy_non_intersection((start, stop), intervals)

            self.intervals = intervals + tuple([(s+start, e+start) for s, e in ai.intervals])
            # self.append_intervals(*ai.intervals)
            # for i_start, i_stop in ai.intervals:
                # intervals = self._union([i_start + start, start + i_stop], intervals)
            # self.intervals = intervals
        else:
            raise NotImplementedError(value)

    @staticmethod
    def _union(interval, intervals):
        start, end = interval
        new_interval = []

        # intervals = np.asarray(intervals)

        for i in intervals:
            i_start, i_end = i
            if (i_start <= end and start <= i_end) or (
                    start <= i_end and i_start <= end):
                start = min(start, i_start)
                end = max(end, i_end)
            else:
                new_interval.append(i)
        new_interval.append((start, end))
        return list(sorted(new_interval))

    @staticmethod
    def _intersection(interval, intervals):
        start, end = interval
        # new_interval = []
        #
        # for i in intervals:
        #     i_start, i_end = i
        #     i_start = max(start, i_start)
        #     i_end = min(end, i_end)
        #     if i_start < i_end:
        #         new_interval.append((i_start, i_end))
        #
        # return list(sorted(new_interval))
        if len(intervals) > 0:
            new_interval = np.array(intervals, copy=True)
            new_interval[:, 0] = np.maximum(start, new_interval[:, 0])
            new_interval[:, 1] = np.minimum(end, new_interval[:, 1])
            return [(s, e) for s, e in new_interval if s < e]
        else:
            return intervals

    @staticmethod
    def _non_intersection(interval, intervals):
        start, end = interval
        new_interval = []

        for i in intervals:
            i_start, i_end = i

            if start < i_start < end:
                i_start = end
            elif start < i_end < end:
                i_end = start
            elif i_start < start and end < i_end:
                new_interval.append((i_start, start))
                i_start = end

            if i_start < i_end:
                new_interval.append((i_start, i_end))

        return list(sorted(new_interval))
        # if len(intervals) > 0:
        #     new_interval = np.array(intervals, copy=True)
        #
        #     index = np.logical_and(start < new_interval[:, 0], new_interval[:, 0] < end)
        #     new_interval[index, 0] = end
        #     index = np.logical_and(start < new_interval[:, 1], new_interval[:, 1] < end)
        #     new_interval[index, 1] = start
        #     index = np.logical_and(new_interval[:, 0] < start, end < new_interval[:, 1])
        #     further = [[i_start, start] for i_start in new_interval[index, 0]]
        #     new_interval[index, 0] = end
        #
        #     return tuple(map(tuple, new_interval.tolist())) + tuple(map(tuple, further))
        # else:
        #     return intervals

    def __getitem__(self, item):
        """

        >>> ai = ArrayIntervall(50)
        >>> ai[19:26]
        array([False, False, False, False, False, False, False])
        >>> ai[10:20] = 1
        >>> ai[25:30] = 1
        >>> ai
        ArrayIntervall("10:20, 25:30", shape=(50,))
        >>> ai[19:26]
        array([ True, False, False, False, False, False,  True])

        """
        start, stop = self._parse_item(item)

        # intervals = self._intersection((start, stop), self.normalized_intervals)
        intervals = cy_intersection((start, stop), self.normalized_intervals)

        arr = np.zeros(stop - start, dtype=np.bool)

        for i_start, i_end in intervals:
            arr[i_start - start:i_end - start] = True

        return arr
