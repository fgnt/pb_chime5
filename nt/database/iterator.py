"""
The reader is part of the new database concept 2017.

The task of the reader is to take a database JSON and an dataset identifier as
an input and load all meta data for each observation with corresponding
numpy arrays for each time signal (non stacked).

An example ID often stands for utterance ID. In case of speaker mixtures,
it replaces mixture ID. Also, in case of event detection utterance ID is not
very adequate.

The JSON file is specified as follows:

datasets:
    <dataset name 0>
        <unique example id 1> (unique within a dataset)
            audio_path:
                speech_source:
                    <path to speech of speaker 0>
                    <path to speech of speaker 1>
                observation:
                    blue_array: (a list, since there are no missing channels)
                        <path to observation of blue_array and channel 0>
                        <path to observation of blue_array and channel 0>
                        ...
                    red_array: (special case for missing channels)
                        c0: <path to observation of red_array and channel 0>
                        c99: <path to observation of red_array and channel 99>
                        ...
                speech_image:
                    ...
            speaker_id:
                <speaker_id for speaker 0>
                ...
            gender:
                <m/f>
                ...
            ...

Make sure, all keys are natsorted in the JSON file.

Make sure, the names are not redundant and it is clear, which is train, dev and
test set. The names should be as close as possible to the original database
names.

An observation/ example has information according to the keys file.

If a database does not have different arrays, the array dimension can be
omitted. Same holds true for the channel axis or the speaker axis.

The different axis have to be natsorted, when they are converted to numpy
arrays. Skipping numbers (i.e. c0, c99) is database specific and is not handled
by a generic implementation.

If audio paths are a list, they will be stacked to a numpy array. If it is a
dictionary, it will become a dictionary of numpy arrays.

If the example IDs are not unique in the original database, the example IDs
are made unique by prefixing them with the dataset name of the original
database, i.e. dt_simu_c0123.
"""
import logging
import numbers
import textwrap
import collections
from collections import ChainMap
from copy import deepcopy
from pathlib import Path

import numpy as np

from nt import kaldi
from nt.database import keys
from nt.io.audioread import audioread

LOG = logging.getLogger('Database')


class BaseIterator:
    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        raise NotImplementedError(
            f'__iter__ is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def __len__(self):
        # The correct exception type is TypeError and not NotImplementedError
        # for __len__. For example len(iterator) ignores TypeError but not
        # NotImplementedError
        raise TypeError(
            f'__len__ is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def __getitem__(self, item):
        if isinstance(item, (slice, tuple, list)):
            return SliceIterator(item, self)
        raise NotImplementedError(
            f'__getitem__ is not implemented for {self.__class__}[{item}],\n'
            f'where type({item}) == {type(item)} '
            f'self: \n{repr(self)}'
        )

    def keys(self):
        raise NotImplementedError(
            f'keys is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def map(self, map_fn):
        """
        :param map_fn: function to transform an example dict. Takes an example
            dict as provided by this iterator and returns a transformed
            example dict, e.g. read and adss the observed audio signals.
        :return: MapIterator returning mapped examples. This can e.g. be
        used to read and add audio to the example dict (see read_audio method).

        Note:
            map_fn can do inplace transformations without using copy.
            The ExampleIterator makes a deepcopy of each example and prevents a
            modification of the root example.
        """
        return MapIterator(map_fn, self)

    def filter(self, filter_fn):
        """
        Filtering examples. If possible this method should be called before
        applying expensive map functions.
        :param filter_fn: function to filter examples, takes example as input
            and returns True if example should be kept, else False.
        :return: FilterIterator iterating over filtered examples.
        """
        return FilterIterator(filter_fn, self)

    def concatenate(self, *others):
        """
        Concatenate this iterator with others. keys need to be unambiguous.
        :param others: list of other iterators to be concatenated
        :return: ExamplesIterator iterating over all examples.
        """
        if len(others) == 0:
            return self
        return ConcatenateIterator(self, *others)

    def zip(self, *others):
        """
        Creates a `Dataset` by zipping together the given datasets.

        This method has two major differences to the built-in `zip()` function
        in Python. First the zipping happen based on the keys of the
        first dataset (i.e. The first defines the order).

        Second it is assumes that all datasets have the same length and keys.
        (Could be removed, when someone needs it.)

        This function is usually followed by a map call to merge the tuple of
        dicts to a single dict.

        >>> ds1 = ExamplesIterator({'a': {'z': 1}, 'b': {'z': 2}})
        >>> ds2 = ExamplesIterator({'a': {'y': 'c'}, 'b': {'y': 'd', 'z': 3}})
        >>> ds3 = ds1.zip(ds2)
        >>> for e in ds3: print(e)
        ({'z': 1, 'example_id': 'a'}, {'y': 'c', 'example_id': 'a'})
        ({'z': 2, 'example_id': 'b'}, {'y': 'd', 'z': 3, 'example_id': 'b'})

        # Merge the dicts, when conflict, prefer the second
        >>> ds4 = ds3.map(lambda example: {**example[0], **example[1]})
        >>> ds4  # doctest: +ELLIPSIS
            ExamplesIterator(len=2)
            ExamplesIterator(len=2)
          ZipIterator()
        MapIterator(<function <lambda> at 0x...>)
        >>> for e in ds4: print(e)
        {'z': 1, 'example_id': 'a', 'y': 'c'}
        {'z': 3, 'example_id': 'b', 'y': 'd'}

        # Lambda that merges an arbitary amount of dicts.
        >>> ds5 = ds3.map(lambda exmaple: dict(sum([list(e.items()) for e in exmaple], [])))
        >>> for e in ds5: print(e)
        {'z': 1, 'example_id': 'a', 'y': 'c'}
        {'z': 3, 'example_id': 'b', 'y': 'd'}

        :param others: list of other iterators to be zipped
        :return: Iterator
        """
        return ZipIterator(self, *others)

    def shuffle(self, reshuffle=False):
        """
        Shuffle this iterator.
        :param reshuffle:
            If True, shuffle on each iteration, but disable indexing.
            If False, single shuffle, but support indexing.
        :return:
        """
        # Should reshuffle default be True or False
        if reshuffle is True:
            return ReShuffleIterator(self)
        elif reshuffle is False:
            return ShuffleIterator(self)
        else:
            raise ValueError(reshuffle, self)

    def split(self, sections):
        """
        >>> examples = {'a': {}, 'b': {}, 'c': {}}
        >>> it = ExamplesIterator(examples)
        >>> its = it.split(2)
        >>> list(its[0])
        [{'example_id': 'a'}, {'example_id': 'b'}]
        >>> list(its[1])
        [{'example_id': 'c'}]
        """
        if sections < 1:
            raise ValueError("sections must be >= 1")
        if sections > len(self):
            raise ValueError(
                f'Iterator has only {len(self)} elements and cannot be '
                f'split into {sections} sections.'
            )
        slices = np.array_split(np.arange(len(self)), sections)
        return [self[list(s)] for s in slices]

    def shard(self, num_shards, shard_index):
        """
        Splits an iterator into `num_shards` shards and
        selects shard `shard_index`.
        """
        return self.split(num_shards)[shard_index]

    def __str__(self):
        return f'{self.__class__.__name__}()'

    def __repr__(self):
        # CB: Discussable, if this methode name should be something like
        #     description instead of __repr__.
        import textwrap
        r = ''
        indent = '  '
        if hasattr(self, 'input_iterator'):
            s = repr(self.input_iterator)
            r += textwrap.indent(s, indent) + '\n'
        if hasattr(self, 'input_iterators'):
            for input_iterator in self.input_iterators:
                s = repr(input_iterator)
                r += textwrap.indent(s, indent) + '\n'
        return r + str(self)


class ExamplesIterator(BaseIterator):
    """
    Iterator to iterate over a list of examples with each example being a dict
    according to the json structure as outline in the top of this file.
    """

    def __init__(self, examples, name=None):
        assert isinstance(examples, dict)
        self.examples = examples
        self.name = name

    def __str__(self):
        if self.name is None:
            return f'{self.__class__.__name__}(len={len(self)})'
        else:
            return f'{self.__class__.__name__}' \
                   f'(name={self.name}, len={len(self)})'

    def keys(self):
        # Note: tuple is immutable, i.e. it can not be modified
        return tuple(self.examples.keys())

    def __iter__(self):
        for k in self.keys():
            yield self[k]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.keys():
                key = item
            else:
                import difflib
                similar = difflib.get_close_matches(item, self.keys())
                raise IndexError(item, f'close_matches: {similar}')
        elif isinstance(item, numbers.Integral):
            key = self.keys()[item]
        else:
            return super().__getitem__(item)
        example = deepcopy(self.examples[key])
        example[keys.EXAMPLE_ID] = key
        return example

    def __len__(self):
        return len(self.examples)


class MapIterator(BaseIterator):
    """
    Iterator that iterates over an input_iterator and applies a transformation
    map_function to each element.

    .. note: This Iterator makes a (deep)copy of the example before applying the
        function.


    """

    def __init__(self, map_function, input_iterator):
        """

        :param map_function: function that transforms an element of
            input_iterator. Use deepcopy within the map_function if necessary.
        :param input_iterator: any iterator (e.g. ExampleIterator)
        """
        assert callable(map_function), map_function
        self.map_function = map_function
        self.input_iterator = input_iterator

    def __str__(self):
        return f'{self.__class__.__name__}({self.map_function})'

    def __len__(self):
        return len(self.input_iterator)

    def __iter__(self):
        for example in self.input_iterator:
            yield self.map_function(example)

    def keys(self):
        return self.input_iterator.keys()

    def __getitem__(self, item):
        if isinstance(item, (str, numbers.Integral)):
            return self.map_function(self.input_iterator[item])
        else:
            return super().__getitem__(item)


class ShuffleIterator(BaseIterator):
    """
    Iterator that shuffles the input_iterator. Assumes, that the input_iterator
    has a length.
    Note:
        This Iterator supports indexing, but does not reshuffle each iteration.

    >>> np.random.seed(1)
    >>> examples = {'a': {}, 'b': {}, 'c': {}}
    >>> it = ExamplesIterator(examples)
    >>> it = it.shuffle(False)
    >>> it
      ExamplesIterator(len=3)
    ShuffleIterator()
    >>> list(it)
    [{'example_id': 'a'}, {'example_id': 'c'}, {'example_id': 'b'}]
    >>> it.keys()
    ('a', 'c', 'b')
    """

    def __init__(self, input_iterator):
        self.permutation = np.arange(len(input_iterator))
        np.random.shuffle(self.permutation)
        self.input_iterator = input_iterator

    def __len__(self):
        return len(self.input_iterator)

    _keys = None

    def keys(self):
        if self._keys is None:
            keys = self.input_iterator.keys()
            self._keys = tuple([keys[p] for p in self.permutation])
        return self._keys

    def __iter__(self):
        for idx in self.permutation:
            yield self.input_iterator[idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_iterator[item]
        elif isinstance(item, numbers.Integral):
            return self.input_iterator[self.permutation[item]]
        else:
            return super().__getitem__(item)


class ReShuffleIterator(BaseIterator):
    """
    Iterator that shuffles the input_iterator. Assumes, that the input_iterator
    has a length.
    Note:
        This Iterator reshuffle each iteration, but does not support indexing.
    """

    def __init__(self, input_iterator):
        self.permutation = np.arange(len(input_iterator))
        self.input_iterator = input_iterator

    def __len__(self):
        return len(self.input_iterator)

    # keys is not well defined for this iterator
    # The First iterator (i.e. ExamplesIterator has sorted keys), so what should
    # this iterator return? Maybe a frozenset to highlight unordered?
    # def keys(self):
    #     return frozenset(self.input_iterator.keys())

    def __iter__(self):
        np.random.shuffle(self.permutation)
        for idx in self.permutation:
            yield self.input_iterator[idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_iterator[item]
        elif isinstance(item, numbers.Integral):
            raise TypeError(item)
        else:
            return super().__getitem__(item)


class SliceIterator(BaseIterator):
    def __init__(self, slice, input_iterator):
        self._slice = slice
        try:
            self.slice = np.arange(len(input_iterator))[self._slice]
        except IndexError:
            if isinstance(slice, (tuple, list)) and isinstance(slice[0], str):
                # Assume sequence of str
                keys = {k: i for i, k in enumerate(input_iterator.keys())}
                self.slice = [keys[k] for k in slice]
            else:
                raise

        self.input_iterator = input_iterator

    def keys(self):
        return self.input_iterator.keys()[self.slice]

    def __len__(self):
        return len(self.slice)

    def __str__(self):
        return f'{self.__class__.__name__}({self._slice})'

    def __iter__(self):
        for idx in self.slice:
            yield self.input_iterator[idx]

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self.input_iterator[self.slice[key]]
        elif isinstance(key, str):
            if key in self.keys():
                return self.input_iterator[key]
            else:
                raise IndexError(key)
        else:
            return super().__getitem__(key)


class FilterIterator(BaseIterator):
    """
    Iterator that iterates only over those elements of input_iterator that meet
    filter_function.
    """

    def __init__(self, filter_function, input_iterator):
        """

        :param filter_function: a function that takes an element of the input
            iterator and returns True if the element is valid else False.
        :param input_iterator: any iterator (e.g. ExampleIterator)
        """
        assert callable(filter_function), filter_function
        self.filter_function = filter_function
        self.input_iterator = input_iterator

    def __str__(self):
        return f'{self.__class__.__name__}({self.filter_function})'

    def __iter__(self):
        for example in self.input_iterator:
            if self.filter_function(example):
                yield example

    def __getitem__(self, key):
        assert isinstance(key, str), (
            f'key == {key}\n{self.__class__} does not support __getitem__ '
            f'for type(key) == {type(key)},\n'
            f'Only type str is allowed.\n'
            f'self:\n{repr(self)}'
        )
        ex = self.input_iterator[key]
        if not self.filter_function(ex):
            raise IndexError(key)
        return ex


class ConcatenateIterator(BaseIterator):
    """
    Iterates over all elements of all input_iterators.
    Best use is to concatenate cross validation or evaluation datasets.
    It does not work well with buffer based shuffle (i.e. in Tensorflow).

    Here, __getitem__ for str is not possible per definition when IDs collide.
    """

    def __init__(self, *input_iterators):
        """
        :param input_iterators: list of iterators
        """
        self.input_iterators = input_iterators

    def __iter__(self):
        for input_iterator in self.input_iterators:
            for example in input_iterator:
                yield example

    def __len__(self):
        return sum([len(i) for i in self.input_iterators])

    _keys = None

    def keys(self):
        """
        >>> examples = {'a': {}, 'b': {}, 'c': {}}
        >>> it = ExamplesIterator(examples)
        >>> it.concatenate(it).keys()
        Traceback (most recent call last):
        ...
        AssertionError: Keys are not unique. There are 3 duplicates.
        ['a', 'b', 'c']
        """
        if self._keys is None:
            keys = []
            for iterator in self.input_iterators:
                keys += list(iterator.keys())
            if len(keys) != len(set(keys)):
                duplicates = [
                    item  # https://stackoverflow.com/a/9835819/5766934
                    for item, count in collections.Counter(keys).items()
                    if count > 1
                ]
                duplicates_str = textwrap.shorten(
                    str(duplicates)[1:-1], width=500, placeholder=' ...')
                raise AssertionError(
                    f'Keys are not unique. '
                    f'There are {len(duplicates)} duplicates.'
                    f'\n[{duplicates_str}]'
                )

            assert len(keys) == len(set(keys)), \
                'Keys are not unique. ' \
                'len(self._keys) = {len(self._keys)} != ' \
                '{len(set(self._keys))} = len(set(self._keys))'
            self._keys = tuple(keys)
        return self._keys

    _chain_map = None

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            for iterator in self.input_iterators:
                if len(iterator) <= item:
                    item -= len(iterator)
                else:
                    return iterator[item]
        elif isinstance(item, str):
            if self._chain_map is None:
                self.keys()  # test unique keys
                self._chain_map = ChainMap(*self.input_iterators)
            return self._chain_map[item]
        else:
            return super().__getitem__(item)


class ZipIterator(BaseIterator):
    """
    See BaseIterator.zip
    """
    def __init__(self, *input_iterators):
        """
        :param input_iterators: list of iterators
        """
        self.input_iterators = input_iterators
        assert len(self.input_iterators) >= 1, \
            'You have to provide at least one iterator.' \
            f'\n{self.input_iterators}'
        assert len(self.input_iterators) >= 2, \
            'Currently limited to at least two iterator. Could be removed.' \
            f'\n{self.input_iterators}'
        lengths = [len(it) for it in self.input_iterators]
        assert len(set(lengths)) == 1, \
            f'Expect that all input_iterators have the same length {lengths}' \
            f'\n{self.input_iterators}'

    def __iter__(self):
        for key in self.keys():
            yield tuple([
                it[key]
                for it in self.input_iterators
            ])

    def __len__(self):
        return len(self.input_iterators[0])

    def keys(self):
        return self.input_iterators[0].keys()

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            item = self.keys()[item]
        if isinstance(item, str):
            return tuple([
                it[item]
                for it in self.input_iterators
            ])
        else:
            return super().__getitem__(item)


class MixIterator(BaseIterator):
    """
    Provide
    """

    def __init__(self, *input_iterators, p=None):
        """
        :param input_iterators:
        :param p: Probabilities for each iterator. Equal probability if None.
        """
        count = len(input_iterators)
        if p is None:
            self.p = np.full((count,), 1 / count)
        else:
            assert count == len(p), f'{count} != {len(p)}'

    def __iter__(self):
        raise NotImplementedError


def recursive_transform(func, dict_list_val, list2array=False):
    """
    Applies a function func to all leaf values in a dict or list or directly to
    a value. The hierarchy of dict_list_val is inherited. Lists are stacked
    to numpy arrays. This function can e.g. be used to recursively apply a
    transformation (e.g. audioread) to all audio paths in an example dict
    (see top of this file).
    :param func: a transformation function to be applied to the leaf values
    :param dict_list_val: dict list or value
    :param args: args for func
    :param kwargs: kwargs for func
    :param list2array:
    :return: dict, list or value with transformed elements
    """
    if isinstance(dict_list_val, dict):
        # Recursively call itself
        return {key: recursive_transform(func, val, list2array)
                for key, val in dict_list_val.items()}
    if isinstance(dict_list_val, (list, tuple)):
        # Recursively call itself
        l = [recursive_transform(func, val, list2array)
             for val in dict_list_val]
        if list2array:
            return np.array(l)
        return l
    else:
        # applies function to a leaf value which is not a dict or list
        return func(dict_list_val)


class AudioReader:
    def __init__(self, src_key='audio_path', dst_key='audio_data',
                 audio_keys='observation', read_fn=lambda x: audioread(x)[0]):
        """
        recursively read audio files and add audio
        signals to the example dict.
        :param src_key: key in an example dict where audio file paths can be
            found.
        :param dst_key: key to add the read audio to the example dict.
        :param audio_keys: str or list of subkeys that are relevant. This can be
            used to prevent unnecessary audioread.
        """
        self.src_key = src_key
        self.dst_key = dst_key
        if audio_keys is not None:
            self.audio_keys = to_list(audio_keys)
        else:
            self.audio_keys = None
        self._read_fn = read_fn

    def __call__(self, example):
        """
        :param example: example dict with src_key in it
        :return: example dict with audio data added
        """
        if self.audio_keys is not None:
            data = {
                audio_key: recursive_transform(
                    self._read_fn, example[self.src_key][audio_key],
                    list2array=True
                )
                for audio_key in self.audio_keys
            }
        else:
            data = recursive_transform(
                self._read_fn, example[self.src_key], list2array=True
            )

        if self.dst_key is not None:
            example[self.dst_key] = data
        else:
            example.update(data)
        return example


class IdFilter:
    def __init__(self, id_list):
        """
        A filter to filter example ids.
        :param id_list: list of valid ids, e.g. ids belonging to a specific
            dataset.

        An alternative with slicing:

        >>> it = ExamplesIterator({'a': {}, 'b': {}, 'c': {}})
        >>> list(it)
        [{'example_id': 'a'}, {'example_id': 'b'}, {'example_id': 'c'}]
        >>> it['a']
        {'example_id': 'a'}
        >>> it['a', 'b']
          ExamplesIterator(len=3)
        SliceIterator(('a', 'b'))
        >>> list(it['a', 'b'])
        [{'example_id': 'a'}, {'example_id': 'b'}]

        >>> it.filter(IdFilter(('a', 'b')))  # doctest: +ELLIPSIS
          ExamplesIterator(len=3)
        FilterIterator(<nt.database.iterator.IdFilter object at 0x...>)
        >>> list(it.filter(IdFilter(('a', 'b'))))
        [{'example_id': 'a'}, {'example_id': 'b'}]
        """
        self.id_list = id_list

    def __call__(self, example):
        """
        :param example: example dict with example_id in it
        :return: True if example_id in id_list else False
        """
        return example[keys.EXAMPLE_ID] in self.id_list


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


class AlignmentReader:
    def __init__(
            self, alignment_path: Path = None, alignments: dict = None,
            example_id_map_fn=lambda x: x[keys.EXAMPLE_ID]):
        assert alignment_path is not None or alignments is not None, (
            'Either alignments or the path to the alignments must be specified'
        )
        self._ali_path = alignment_path
        self._alignments = alignments
        self._map_fn = example_id_map_fn

    def __call__(self, example):
        if self._alignments is None:
            self._alignments = \
                kaldi.alignment.import_alignment_data(self._ali_path)
            LOG.debug(
                f'Read {len(self._alignments)} alignments '
                f'from path {self._ali_path}'
            )
        try:
            example[keys.ALIGNMENT] = self._alignments[
                self._map_fn(example)
            ]
            example[keys.NUM_ALIGNMENT_FRAMES] = len(example[keys.ALIGNMENT])
        except KeyError:
            LOG.warning(
                f'No alignment found for example id {example[keys.EXAMPLE_ID]} '
                f'(mapped: {self._map_fn(example)}).'
            )
        return example


def remove_examples_without_alignment(example):
    valid_ali = keys.ALIGNMENT in example and len(example[keys.ALIGNMENT])
    if not valid_ali:
        LOG.warning(f'No alignment found for example\n{example}')
        return False
    if keys.NUM_SAMPLES in example:
        num_samples = example[keys.NUM_SAMPLES]
        if isinstance(num_samples, dict):
            num_samples = num_samples[keys.OBSERVATION]
    else:
        return True  # Only happens for Kaldi databases
    num_frames = (num_samples - 400 + 160) // 160
    num_frames_lfr = (num_frames + np.mod(-num_frames, 3)) // 3
    len_ali = len(example[keys.ALIGNMENT])
    valid_ali = (
        len_ali == num_frames or
        len_ali == num_frames_lfr
    )
    if not valid_ali:
        LOG.warning(
            f'Alignment has {len_ali} frames but the observation has '
            f'{num_frames} [{num_frames_lfr}] frames. Example was:\n{example}'
        )
        return False
    return True


def remove_zero_length_example(example, audio_key='observation',
                                dst_key='audio_data'):

    if keys.NUM_SAMPLES in example:
        num_samples = example[keys.NUM_SAMPLES]
        if isinstance(num_samples, dict):
            num_samples = num_samples[keys.OBSERVATION]
        valid_ali = num_samples > 0
    else:
        valid_ali = len(example[dst_key][audio_key]) > 0
    if not valid_ali:
        LOG.warning(f'Skipping: Audio length '
                    f'example\n{example[keys.EXAMPLE_ID]} is 0')
        return False
    return True


class LimitAudioLength:
    def __init__(self, max_lengths=160000, audio_key='observation',
                 dst_key='audio_data', frame_length=400, frame_step=160):
        self.max_lengths = max_lengths
        self.audio_key = audio_key
        self.dst_key = dst_key
        self.frame_length = frame_length
        self.frame_step = frame_step

    def __call__(self, example):
        valid_ex = keys.NUM_SAMPLES in example and \
                   example[keys.NUM_SAMPLES] <= self.max_lengths
        if not valid_ex:
            delta = max(1, (example[keys.NUM_SAMPLES] - self.max_lengths) // 2)
            start = np.random.choice(delta, 1)[0]

            # audio
            example[self.dst_key][self.audio_key] = \
                example[self.dst_key][self.audio_key][start: start
                + self.max_lengths]
            example[keys.NUM_SAMPLES] = self.max_lengths

            # alignment
            num_frames_start = max(
                            0, (start - self.frame_length + self.frame_step)
                            // self.frame_step)
            num_frames_length = \
                (self.max_lengths - self.frame_length + self.frame_step)\
                // self.frame_step
            example[keys.ALIGNMENT] = \
                example[keys.ALIGNMENT][num_frames_start: num_frames_start
                + num_frames_length]
            example[keys.NUM_ALIGNMENT_FRAMES] = num_frames_length

            LOG.warning(f'Cutting example to length {self.max_lengths}'
                        f' :\n{example[keys.EXAMPLE_ID]}')
        return example


class Word2Id:
    def __init__(self, word2id_fn):
        self._word2id_fn = word2id_fn

    def __call__(self, example):
        def _w2id(s):
            return np.array([self._word2id_fn(w) for w in s.split()], np.int32)

        for trans in [keys.TRANSCRIPTION, keys.KALDI_TRANSCRIPTION]:
            try:
                example[trans + '_ids'] = recursive_transform(
                    _w2id, example[trans]
                )
            except KeyError:
                pass
        return example
