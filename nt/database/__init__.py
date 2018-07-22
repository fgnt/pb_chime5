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
import glob
import logging
import pickle
from collections import defaultdict
from pathlib import Path
import weakref
from cached_property import cached_property

import numpy as np

from nt import kaldi
from nt.io import load_json
from nt.io.audioread import audioread

from nt.database import keys
from nt.database.iterator import BaseIterator
from nt.database.iterator import ExamplesIterator

LOG = logging.getLogger('Database')


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


class DictDatabase:
    def __init__(self, database_dict: dict):
        """

        :param json_path: path to database JSON
        """
        self.database_dict = database_dict

    @property
    def dataset_names(self):
        return list(self.database_dict[keys.DATASETS].keys())

    def get_iterator_by_names(self, dataset_names):
        """
        Returns a single Iterator over specified datasets.

        Adds the example_id and dataset_name to each example dict.

        :param dataset_names: list or str specifying the datasets of interest.
            If None an iterator over the complete databases will be returned.
        :return:
        """
        dataset_names = to_list(dataset_names)
        iterators = list()
        for dataset_name in dataset_names:
            try:
                examples = self.database_dict[keys.DATASETS][dataset_name]
            except KeyError:
                import difflib
                similar = difflib.get_close_matches(
                    dataset_name,
                    self.database_dict[keys.DATASETS].keys(),
                    n=5,
                    cutoff=0,
                )
                raise KeyError(dataset_name, f'close_matches: {similar}', self)

            for example_id in examples.keys():
                examples[example_id][keys.EXAMPLE_ID] = example_id
                examples[example_id][keys.DATASET_NAME] = dataset_name

            # Convert values to binary, because deepcopy on binary is faster
            # This is important for CHiME5
            examples = {k: pickle.dumps(v) for k, v in examples.items()}
            it = ExamplesIterator(examples, name=dataset_name)
            # Apply map function to restore binary data
            it = it.map(pickle.loads)
            iterators.append(it)

        return BaseIterator.concatenate(*iterators)

    def get_lengths(self, datasets, length_transform_fn=lambda x: x):
        raise NotImplementedError

    def get_bucket_boundaries(
            self, datasets, num_buckets=1, length_transform_fn=lambda x: x
    ):
        try:
            lengths = self.get_lengths(datasets, length_transform_fn)
            lengths_list = [length for length in lengths.values()]
            percentiles = np.linspace(
                0, 100, num_buckets + 1, endpoint=True)[1:-1]
            return np.percentile(lengths_list, percentiles,
                                 interpolation='higher')
        except NotImplementedError:
            assert num_buckets == 1, num_buckets
            return []

    @property
    def audio_read_fn(self):
        return lambda x: audioread(x)[0]


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
        it = self.get_iterator_by_names(datasets)
        lengths = dict()
        for example in it:
            num_samples = example[keys.NUM_SAMPLES]
            if isinstance(num_samples, dict):
                num_samples = num_samples[keys.OBSERVATION]
            example_id = example[keys.EXAMPLE_ID]
            lengths[example_id] = (length_transform_fn(num_samples))
        return lengths

    def add_num_samples(self, example):
        return example


class KaldiDatabase(DictDatabase):

    def __init__(self, egs_path: Path):
        if isinstance(egs_path, str):
            egs_path = Path(egs_path)
        self._egs_path = egs_path
        super().__init__(self.get_dataset_dict_from_kaldi(egs_path))

    def __repr__(self):
        return f'{type(self).__name__}: {self._egs_path}'

    @staticmethod
    def get_examples_from_dataset(dataset_path):
        scp = kaldi.io.read_keyed_text_file(dataset_path / 'wav.scp')
        utt2spk = kaldi.io.read_keyed_text_file(dataset_path / 'utt2spk')
        text = kaldi.io.read_keyed_text_file(dataset_path / 'text')
        try:
            spk2gender = kaldi.io.read_keyed_text_file(dataset_path / 'spk2gender')
        except FileNotFoundError:
            spk2gender = None
        examples = dict()

        # Normally the scp points to a single audio file (i.e. len(s) = 1)
        # For databases with a different audio formart (e.g. WSJ) however,
        # it is a command to convert the corresponding audio file. The
        # file is usually at the end of this command. If this does not work,
        # additional heuristics need to be introduced here.
        def _audio_path(s):
            if len(s) == 1:
                return s[0]
            else:
                return s[-2]

        for example_id in scp:
            example = defaultdict(dict)
            example[keys.AUDIO_PATH][keys.OBSERVATION] = _audio_path(scp[example_id])
            example[keys.SPEAKER_ID] = utt2spk[example_id][0]
            if spk2gender is not None:
                example[keys.GENDER] = spk2gender[example[keys.SPEAKER_ID]][0]
            example[keys.KALDI_TRANSCRIPTION] = ' '.join(text[example_id])
            examples[example_id] = dict(**example)
        return examples

    def add_num_samples(self, example):
        assert (
            keys.AUDIO_DATA in example
            and keys.OBSERVATION in example[keys.AUDIO_DATA]
        ), (
            'No audio data found in example. Make sure to map with '
            '`AudioReader` before adding `num_samples`.'
        )
        example[keys.NUM_SAMPLES] \
            = example[keys.AUDIO_DATA][keys.OBSERVATION].shape[-1]
        return example

    @classmethod
    def get_dataset_dict_from_kaldi(cls, egs_path):
        scp_paths = glob.glob(str(egs_path / 'data' / '*' / 'wav.scp'))
        dataset_dict = {'datasets': {}}
        for wav_scp_file in scp_paths:
            dataset_path = Path(wav_scp_file).parent
            dataset_name = dataset_path.name
            examples = cls.get_examples_from_dataset(dataset_path)
            dataset_dict['datasets'][dataset_name] = examples
        return dataset_dict


class HybridASRDatabaseTemplate:

    def __init__(self, lfr=False):
        self.lfr = lfr

    @property
    def datasets_train(self):
        """A list of filelist names for training."""
        raise NotImplementedError

    @property
    def datasets_eval(self):
        """A list of filelist names for development."""
        raise NotImplementedError

    @property
    def datasets_test(self):
        """A list of filelist names for evaluation."""
        raise NotImplementedError

    @property
    def ali_path_train(self):
        """Path containing the kaldi alignments for train data."""
        if self.lfr:
            return self.ali_path_train_lfr
        else:
            return self.ali_path_train_ffr

    @property
    def ali_path_train_ffr(self):
        """Path containing the kaldi alignments for train data."""
        raise NotImplementedError

    @property
    def ali_path_train_lfr(self):
        """Path containing the kaldi alignments for train data."""
        raise NotImplementedError

    @property
    def ali_path_eval(self):
        """Path containing the kaldi alignments for dev data."""
        if self.lfr:
            return self.ali_path_eval_lfr
        else:
            return self.ali_path_eval_ffr

    @property
    def ali_path_eval_ffr(self):
        """Path containing the kaldi alignments for dev data."""
        raise NotImplementedError

    @property
    def ali_path_eval_lfr(self):
        """Path containing the kaldi alignments for dev data."""
        raise NotImplementedError

    @property
    def hclg_path(self):
        """Path to HCLG directory created by Kaldi."""
        if self.lfr:
            return self.hclg_path_lfr
        else:
            return self.hclg_path_ffr

    @property
    def lang_path(self):
        raise NotImplementedError

    @property
    def hclg_path_ffr(self):
        """Path to HCLG directory created by Kaldi."""
        raise NotImplementedError

    @property
    def hclg_path_lfr(self):
        """Path to HCLG directory created by Kaldi."""
        return self.ali_path_train_lfr / 'graph_tgpr_5k'

    @property
    def example_id_map_fn(self):
        return lambda x: x[keys.EXAMPLE_ID]

    @property
    def audio_read_fn(self):
        return lambda x: audioread(x)[0]

    @property
    def decode_fst(self):
        """A string pointing to HCLG.fst from the kaldi recipe."""
        return str(self.hclg_path / 'HCLG.fst')

    @property
    def words_txt(self):
        """A string pointing to the `words.txt` created by Kaldi."""
        return str(self.hclg_path / 'words.txt')

    @property
    def model_file(self):
        return str(self.ali_path_train / 'final.mdl')

    @property
    def tree_file(self):
        return str(self.ali_path_train / 'tree')

    @property
    def phones(self):
        return str(self.ali_path_train / 'phones.txt')

    @property
    def occs_file(self):
        if self.lfr:
            return self.occs_file_lfr
        else:
            return self.occs_file_ffr

    @property
    def occs_file_ffr(self):
        return str(self.ali_path_train_ffr / 'final.occs')

    @property
    def occs_file_lfr(self):
        return str(self.ali_path_train_lfr / '1.occs')

    @cached_property
    def occs(self):
        """An array with the number of occurances for each state."""
        return kaldi.alignment.import_occs(self.occs_file)

    @cached_property
    def _id2word_dict(self):
        return kaldi.io.id2word(self.words_txt)

    @cached_property
    def _word2id_dict(self):
        return kaldi.io.word2id(self.words_txt)

    @cached_property
    def _phone2id_dict(self):
        return kaldi.io.read_keyed_text_file(self.phones)

    @cached_property
    def _id2phone_dict(self):
        return {int(v[0]): k for k, v in self._phone2id_dict.items()}

    def phone2id(self, phone):
        return self._phone2id_dict[phone]

    def id2phone(self, id_):
        return self._id2phone_dict[id_]

    def word2id(self, word):
        """Returns the integer ID for a given word.

        If the word is not found, it returns the ID for `<UNK>`.
        """
        try:
            return self._word2id_dict[word]
        except KeyError:
            return self._word2id_dict['<UNK>']

    def id2word(self, _id):
        """Returns the word corresponding to `_id`."""
        return self._id2word_dict[_id]

    def get_length_for_dataset(self, dataset):
        return len(self.get_iterator_by_names(dataset))

    def write_text_file(self, filename, datasets):
        iterator = self.get_iterator_by_names(datasets)
        with open(filename, 'w') as fid:
            for example in iterator:
                fid.write(
                    f'{example[keys.EXAMPLE_ID]} '
                    f'{example[keys.KALDI_TRANSCRIPTION]}\n'
                )

    def utterances_for_dataset(self, dataset):
        iterator = self.get_iterator_by_names(dataset)
        return [ex[keys.EXAMPLE_ID] for ex in iterator]

    @cached_property
    def state_alignment(self):
        alignments = kaldi.alignment.import_alignment_data(
            self.ali_path_train, model_name=self.model_file
        )
        alignments.update(kaldi.alignment.import_alignment_data(
            self.ali_path_eval, model_name=self.model_file
        ))
        return alignments

    @cached_property
    def phone_alignment(self):
        alignments = kaldi.alignment.import_alignment_data(
            self.ali_path_train,
            import_fn=kaldi.alignment.import_phone_alignment_from_file,
            per_frame=True, model_name=self.model_file
        )
        alignments.update(kaldi.alignment.import_alignment_data(
            self.ali_path_eval,
            import_fn=kaldi.alignment.import_phone_alignment_from_file,
            per_frame=True, model_name=self.model_file
        ))
        return alignments

    @cached_property
    def vad(self):
        alignment = self.phone_alignment
        with open(self.lang_path / 'phones' / 'silence.csl') as fid:
            silence_ids = list(map(int, fid.read().strip().split(':')))
        return {
            k: np.asarray([int(_id) not in silence_ids for _id in v])
            for k, v in alignment.items()
        }

    @property
    def asr_observation_key(self):
        return keys.OBSERVATION

    def build_select_channels_map_fn(self, channels):
        def select_channels(example):
            assert channels == [0], (
                f'Requested to select channels {channels}, but the '
                f'database is only single-channel. Please only request '
                f'channel 0 in this case (channels = [0]).'
            )
            return example
        return select_channels

    def build_sample_channels_map_fn(self, channels):
        def sample_channels(example):
            assert channels == [0], (
                f'Requested to sample from channels {channels}, but the '
                f'database is only single-channel. Please only request '
                f'channel 0 in this case (channels = [0]).'
            )
            return example
        return sample_channels


class HybridASRJSONDatabaseTemplate(HybridASRDatabaseTemplate, JsonDatabase):
    def __init__(self, json_path: Path, lfr=False):
        super().__init__(lfr=lfr)
        super(HybridASRDatabaseTemplate, self).__init__(json_path=json_path)


class HybridASRKaldiDatabaseTemplate(HybridASRDatabaseTemplate, KaldiDatabase):
    def __init__(self, egs_path: Path, lfr=False):
        super().__init__(lfr=lfr)
        super(HybridASRDatabaseTemplate, self).__init__(egs_path=egs_path)
