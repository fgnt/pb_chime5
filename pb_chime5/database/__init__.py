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
from pathlib import Path
import weakref
from cached_property import cached_property

from pb_chime5.io import load_json
from pb_chime5.io.audioread import load_audio

from pb_chime5.database.keys import *
import lazy_dataset

LOG = logging.getLogger('Database')


def to_list(x, item_type=None):
    """
    Note:
        It is recommended to use item_type, when the type of the list is known
        to catch as much cases as possible.
        The problem is that many python functions return a type that does not
        inherit from tuple and/or list.
        e.g. dict keys, dict values, map, sorted, ...

        The instance check with collections.Sequence could produce problem with
        str. (isinstance('any str', collections.Sequence) is True)

    >>> to_list(1)
    [1]
    >>> to_list([1])
    [1]
    >>> to_list((1,))
    (1,)
    >>> to_list({1: 2}.keys())  # Wrong
    [dict_keys([1])]
    >>> to_list({1: 2}.keys(), item_type=int)
    [1]
    """
    if item_type is None:
        if isinstance(x, (list, tuple)):
            return x
        return [x]
    else:
        if isinstance(x, item_type):
            return [x]
        return list(x)


class DictDatabase:
    def __init__(self, database_dict: dict):
        """

        :param json_path: path to database JSON
        """
        self.database_dict = database_dict

    @property
    def dataset_names(self):
        return tuple(
            self.database_dict[DATASETS].keys()
        ) + tuple(
            self.database_dict.get(ALIAS, {}).keys()
        )

    @property
    def datasets_train(self):
        """A list of filelist names for training."""
        raise NotImplementedError

    @property
    def datasets_eval(self):
        """A list of filelist names for evaluation."""
        raise NotImplementedError

    @property
    def datasets_test(self):
        """A list of filelist names for testing."""
        raise NotImplementedError

    @cached_property
    def datasets(self):
        dataset_names = self.dataset_names
        return type(
            'DatasetsCollection',
            (object,),
            {
                # 'abc': property(lambda self: 'cdf'),
                '__getitem__': (lambda _, dataset_name: self.get_datasets(dataset_name)),
                **{
                    k: property(lambda self: self[k])
                    for k in dataset_names
                }
            }
        )()

    def _get_dataset_from_database_dict(self, dataset_name):
        if dataset_name in self.database_dict.get('alias', []):
            dataset_names = self.database_dict['alias'][dataset_name]
            examples = {}
            for name in dataset_names:
                examples_new = self.database_dict[DATASETS][name]
                intersection = set.intersection(
                    set(examples.keys()),
                    set(examples_new.keys()),
                )
                assert len(intersection) == 0, intersection
                examples = {**examples, **examples_new}
            return examples
        else:
            return self.database_dict[DATASETS][dataset_name]

    @cached_property
    def _iterator_weak_ref_dict(self):
        return weakref.WeakValueDictionary()

    def get_datasets(self, dataset_names, use_weakref=True):
        """
        Returns a single Iterator over specified datasets.

        Adds the example_id and dataset_name to each example dict.

        :param dataset_names: list or str specifying the datasets of interest.
            If None an iterator over the complete databases will be returned.
        :return:
        """
        dataset_names = to_list(dataset_names, item_type=str)
        iterators = list()
        for dataset_name in dataset_names:
            if use_weakref:
                try:
                    it = self._iterator_weak_ref_dict[dataset_name]
                except KeyError:
                    pass
                else:
                    iterators.append(it)
                    continue
            try:
                examples = self._get_dataset_from_database_dict(dataset_name)
            except KeyError:
                import difflib
                similar = difflib.get_close_matches(
                    dataset_name,
                    self.dataset_names,
                    n=5,
                    cutoff=0,
                )
                raise KeyError(dataset_name, f'close_matches: {similar}', self)
            if len(examples) == 0:
                # When somebody need empty datasets, add an option to this
                # function to allow empty datasets.
                raise RuntimeError(
                    f'The requested dataset {dataset_name!r} is empty. '
                )

            for example_id in examples.keys():
                examples[example_id][EXAMPLE_ID] = example_id
                examples[example_id][DATASET_NAME] = dataset_name

            # Convert values to binary, because deepcopy on binary is faster
            # This is important for CHiME5
            ds = lazy_dataset.from_dict(examples)

            if use_weakref:
                self._iterator_weak_ref_dict[dataset_name] = ds

            iterators.append(ds)

        return lazy_dataset.concatenate(*iterators)

    @property
    def read_fn(self):
        return lambda x: load_audio(x)[0]


class JsonDatabase(DictDatabase):
    def __init__(self, json_path: [str, Path]):
        """

        :param json_path: path to database JSON
        """
        self._json_path = json_path

    @cached_property
    def database_dict(self):
        LOG.info(f'Using json {self._json_path}')
        return load_json(self._json_path)

    def __repr__(self):
        return f'{type(self).__name__}({self._json_path!r})'

    def get_lengths(self, datasets, length_transform_fn=lambda x: x):
        it = self.get_datasets(datasets)
        lengths = dict()
        for example in it:
            num_samples = example[NUM_SAMPLES]
            if isinstance(num_samples, dict):
                num_samples = num_samples[OBSERVATION]
            example_id = example[EXAMPLE_ID]
            lengths[example_id] = (length_transform_fn(num_samples))
        return lengths

    def add_num_samples(self, example):
        return example
