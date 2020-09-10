from pathlib import Path
import collections
import itertools
from cached_property import cached_property
from lazy_dataset.database import Database

import numpy as np
import paderbox as pb


def groupby(iterable, group_fn: callable, map_fn=None):
    groups = collections.defaultdict(list)
    for k, g in itertools.groupby(iterable, group_fn):
        if map_fn is None:
            groups[k].extend(g)
        else:
            groups[k].extend(map(map_fn, g))
    return dict(groups)


def get_chime6_files(chime6_dir, worn=False, flat=False):
    """
    >>> from paderbox.notebook import pprint
    >>> chime6_dir = Path('/net/fastdb/chime6/CHiME6')
    >>> pprint(get_chime6_files(chime6_dir, worn=True)['S02'])  # doctest: +ELLIPSIS
    {'P05': '.../CHiME6/audio/dev/S02_P05.wav',
     'P06': '.../CHiME6/audio/dev/S02_P06.wav',
     'P07': '.../CHiME6/audio/dev/S02_P07.wav',
     'P08': '.../CHiME6/audio/dev/S02_P08.wav'}
    >>> pprint(get_chime6_files(chime6_dir, worn=False)['S02'])  # doctest: +ELLIPSIS
    {'U01': ['.../CHiME6/audio/dev/S02_U01.CH1.wav',
      '.../CHiME6/audio/dev/S02_U01.CH2.wav',
      '.../CHiME6/audio/dev/S02_U01.CH3.wav',
      '.../CHiME6/audio/dev/S02_U01.CH4.wav'],
     'U02': ['.../CHiME6/audio/dev/S02_U02.CH1.wav',
      '.../CHiME6/audio/dev/S02_U02.CH2.wav',
      '.../CHiME6/audio/dev/S02_U02.CH3.wav',
      '.../CHiME6/audio/dev/S02_U02.CH4.wav'],
     'U03': ['.../CHiME6/audio/dev/S02_U03.CH1.wav',
      '.../CHiME6/audio/dev/S02_U03.CH2.wav',
      '.../CHiME6/audio/dev/S02_U03.CH3.wav',
      '.../CHiME6/audio/dev/S02_U03.CH4.wav'],
     'U04': ['.../CHiME6/audio/dev/S02_U04.CH1.wav',
      '.../CHiME6/audio/dev/S02_U04.CH2.wav',
      '.../CHiME6/audio/dev/S02_U04.CH3.wav',
      '.../CHiME6/audio/dev/S02_U04.CH4.wav'],
     'U05': ['.../CHiME6/audio/dev/S02_U05.CH1.wav',
      '.../CHiME6/audio/dev/S02_U05.CH2.wav',
      '.../CHiME6/audio/dev/S02_U05.CH3.wav',
      '.../CHiME6/audio/dev/S02_U05.CH4.wav'],
     'U06': ['.../CHiME6/audio/dev/S02_U06.CH1.wav',
      '.../CHiME6/audio/dev/S02_U06.CH2.wav',
      '.../CHiME6/audio/dev/S02_U06.CH3.wav',
      '.../CHiME6/audio/dev/S02_U06.CH4.wav']}
    >>> pprint(get_chime6_files(chime6_dir, worn=False, flat=True)['S02'])  # doctest: +ELLIPSIS
    ['/net/fastdb/chime6/CHiME6/audio/dev/S02_U01.CH1.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U01.CH2.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U01.CH3.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U01.CH4.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U02.CH1.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U02.CH2.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U02.CH3.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U02.CH4.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U03.CH1.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U03.CH2.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U03.CH3.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U03.CH4.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U04.CH1.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U04.CH2.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U04.CH3.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U04.CH4.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U05.CH1.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U05.CH2.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U05.CH3.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U05.CH4.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U06.CH1.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U06.CH2.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U06.CH3.wav',
     '/net/fastdb/chime6/CHiME6/audio/dev/S02_U06.CH4.wav']
    """
    if worn:
        assert flat is False, flat
        files = sorted(Path(chime6_dir).glob('audio/*/*_P*.wav'))
        grouped_files = {
            (p.name.split('_')[0], p.name.split('_')[1].split('.')[0]): str(p)
            for p in files
        }
        assert len(files) == len(grouped_files), (files, grouped_files)
    else:
        files = sorted(Path(chime6_dir).glob('audio/*/*_U*.wav'))
        if flat:
            grouped_files = groupby(
                [
                    (
                        (p.name.split('_')[0],),
                        str(p)
                    )
                    for p in files
                ],
                group_fn=lambda e: e[0],
                map_fn=lambda e: e[1]
            )
        else:
            grouped_files = groupby(
                [
                    (
                        (p.name.split('_')[0], p.name.split('_')[1].split('.')[0]),
                        str(p)
                    )
                    for p in files
                ],
                group_fn=lambda e: e[0],
                map_fn=lambda e: e[1]
            )
            assert len(files) == len(grouped_files) * 4, (files, grouped_files)
    assert len(files) > 0, (files, chime6_dir)
    return pb.utils.nested.deflatten(grouped_files, sep=None)


class Chime6RTTMDatabase(Database):
    def __init__(self, rttm_path, chime6_dir):
        super().__init__()
        self._rttm_path = rttm_path
        self._chime6_dir = chime6_dir

    @cached_property
    def _rttm(self):
        from paderbox.array import intervall as array_intervall
        rttm = array_intervall.from_rttm(self._rttm_path)

        original_keys = tuple(rttm.keys())
        
        # The default scripts have a strange convention and add some postfixes
        # that have to be removed. e.g.:
        # S02_U06.ENH or S02_U06
        rttm = {
            k.replace('_U06', '').replace('.ENH', ''): v
            for k, v in rttm.items()
        }
        assert len(rttm.keys()) == len(original_keys), (rttm.keys(), original_keys)

        return rttm

    @cached_property
    def _array_files(self):
        return get_chime6_files(self._chime6_dir)

    @cached_property
    def _worn_files(self):
        return get_chime6_files(self._chime6_dir, worn=True)

    @cached_property
    def _alias(self):
        """
        >>> from paderbox.notebook import pprint
        >>> from pathlib import Path
        >>> from pb_chime5.database.chime5.rttm import Chime6RTTMDatabase
        >>> chime6_dir = Path('/net/fastdb/chime6/CHiME6')
        >>> pprint(RTTMDatabase(None, chime6_dir)._alias)
        {'dev': ['S02', 'S09'],
         'eval': ['S01', 'S21'],
         'train': ['S03',
          'S04',
          'S05',
          'S06',
          'S07',
          'S08',
          'S12',
          'S13',
          'S16',
          'S17',
          'S18',
          'S19',
          'S20',
          'S22',
          'S23',
          'S24']}
        """
        groups = groupby(
            sorted(self._chime6_dir.glob('audio/*/*.wav')),
            group_fn=lambda path: path.parts[-2],
            map_fn=lambda path: path.name.split('_')[0]
        )
        return {
            k: sorted(set(v))
            for k, v in groups.items()
        }


    @cached_property
    def data(self):
        datasets = {}
        max_samples = len(str(16000 * 60 * 60 * 10))  # 10h
        for session_id, session in self._rttm.items():
            datasets[session_id] = {}
            for speaker_id, speaker in session.items():
                for start, end in speaker.intervals:
                    # use zfill to ensure correct sorting later

                    # local/get_hyp_perspeaker_perarray_file.py expects:
                    #     uttid_id = 'S09_U06.ENH-4-704588-704738'
                    #     micid = uttid_id.strip().split('.')[0].strip().split('_')[1]
                    #     speakerid = uttid_id.strip().split('-')[1]
                    #     sessionid = uttid_id.strip().split('_')[0]

                    # The kaldi baseline can only be executed, when the
                    # utterance ID contains U06, independently is this array
                    # was used.
                    example_id = f'{session_id}_U06.-{speaker_id}-{str(start).zfill(max_samples)}_{str(end).zfill(max_samples)}'
                    datasets[session_id][example_id] = {
                        'example_id': example_id,
                        'start': start,
                        'end': end,
                        'num_samples': end - start,
                        'session_id': session_id,
                        'speaker_id': speaker_id,
                        'audio_path': {
                            'observation': self._array_files[session_id],
                            'worn': self._worn_files[session_id],
                        }
                    }
        return {
            'datasets': datasets,
            'alias': self._alias,
        }

    def get_iterator_for_session(
            self,
            session,
            *,
            audio_read=False,
            drop_unknown_target_speaker=False,
            adjust_times=False,
            context_samples=0,
            equal_start_context=False,
    ):
        from pb_chime5.database.chime5.database import (
            backup_orig_start_end,
            adjust_start_end,
            AddContext,
            Chime5AudioReader,
        )
            
        if isinstance(session, str):
            session = (session, )

        it = self.get_dataset(session)

        # # Ignore drop_unknown_target_speaker
        # if drop_unknown_target_speaker:
        #     it = it.filter(lambda ex: ex['transcription'] != '[redacted]', lazy=False)

        if context_samples is not 0 or adjust_times:
            it = it.map(backup_orig_start_end)

        if adjust_times:
            if adjust_times is True:
                assert drop_unknown_target_speaker, (
                    'adjust_times is undefined for '
                    'ex["target_speaker"] == "unknown". '
                    'Set adjust_times to True.'
                )
                it = it.map(adjust_start_end)
            else:
                raise ValueError(adjust_times)

        if context_samples is not 0:
            # adjust_times should be before AddContext, because AddContext
            # adds the new key start_context that
            it = it.map(AddContext(
                context_samples,
                equal_start_context=equal_start_context,
            ))

        if audio_read is False:
            pass
        elif audio_read is True:
            it = it.map(Chime5AudioReader(audio_keys=None))
        else:
            raise TypeError(audio_read)

        return it


class RTTMDatabase(Database):
    def __init__(self, rttm_path, audio_paths, alias=None):
        """
        A database that is generated from a rttm file and a dict of the audio
        files.

        Args:
            rttm_path:
                str, path or list of str/path to rttm files.
            audio_paths:
                dict:
                    key: file_id, e.g. S02
                    item: files, e.g. all array files from S02_U01.CH1.wav to S02_U06.CH4.wav
            alias:
                dict, e.g. {'dev': ['S02', 'S09'], ...}

        >>> from paderbox.utils.pretty import pprint
    
        >>> chime6_dir = Path('/net/fastdb/chime6/CHiME6')
        >>> chime6_rttm = Path('/net/vol/boeddeker/chime6/kaldi/egs/chime6/chime6_rttm')
    
        >>> rttm = sorted(chime6_rttm.glob('*_rttm'))
    
        >>> audio_paths = get_chime6_files(chime6_dir, worn=False, flat=True)
        >>> audio_paths = {k: [p for p in paths if 'CH1' in p] for k, paths in audio_paths.items()}
        >>> alias = chime6_dir.glob('transcriptions/*/*.json')
        >>> alias = groupby(sorted(alias), lambda p: p.parts[-2], lambda p: p.with_suffix('').name)
        >>> pprint(alias)
        {'dev': ['S02', 'S09'],
         'eval': ['S01', 'S21'],
         'train': ['S03',
          'S04',
          'S05',
          'S06',
          'S07',
          'S08',
          'S12',
          'S13',
          'S16',
          'S17',
          'S18',
          'S19',
          'S20',
          'S22',
          'S23',
          'S24']}
    
        >>> db = RTTMDatabase(rttm, audio_paths, alias)
        >>> pprint(db.dataset_names)
        ('S09',
         'S02',
         'S21',
         'S01',
         'S03',
         'S04',
         'S05',
         'S06',
         'S07',
         'S17',
         'S08',
         'S16',
         'S12',
         'S13',
         'S18',
         'S22',
         'S19',
         'S20',
         'S23',
         'S24',
         'dev',
         'eval',
         'train')
    
        # U05 will be dropped, because it is 15 minitues shorter and does not have
        # audio for this file.
        >>> ds = db.get_dataset_for_session('S12', audio_read=True, equal_start_context=True, context_samples = 400 * 16000)
        >>> pprint(ds[-1])
        {'example_id': 'S12_U06.-P36-131931840_131941760',
         'start': 125531840,
         'end': 138341760,
         'num_samples': 12809920,
         'session_id': 'S12',
         'speaker_id': 'P36',
         'audio_path': ['/net/fastdb/chime6/CHiME6/audio/train/S12_U01.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/train/S12_U02.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/train/S12_U03.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/train/S12_U04.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/train/S12_U05.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/train/S12_U06.CH1.wav'],
         'dataset': 'S12',
         'start_orig': 131931840,
         'end_orig': 131941760,
         'num_samples_orig': 9920,
         'audio_data': array(shape=(5, 12809920), dtype=float64)}
    
        # The length of all audio files is different.
        # Test that the loaded length is the minimum length, i.e. drop the last samples for some audio files.
        >>> ds = db.get_dataset_for_session('S02', audio_read=True, equal_start_context=True, context_samples = (0, 400 * 16000))
        >>> pprint(ds[-1])
        {'example_id': 'S02_U06.-P05-142186400_142202560',
         'start': 142186400,
         'end': 148602560,
         'num_samples': 6416160,
         'session_id': 'S02',
         'speaker_id': 'P05',
         'audio_path': ['/net/fastdb/chime6/CHiME6/audio/dev/S02_U01.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/dev/S02_U02.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/dev/S02_U03.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/dev/S02_U04.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/dev/S02_U05.CH1.wav',
          '/net/fastdb/chime6/CHiME6/audio/dev/S02_U06.CH1.wav'],
         'dataset': 'S02',
         'start_orig': 142186400,
         'end_orig': 142202560,
         'num_samples_orig': 16160,
         'audio_data': array(shape=(6, 285847), dtype=float64)}
        """
        super().__init__()
        self._rttm_path = rttm_path
        self._audio_paths = audio_paths
        if alias is None:
            alias = []
        self._alias = alias

    @cached_property
    def _rttm(self):
        from paderbox.array import intervall as array_intervall
        rttm = array_intervall.from_rttm(self._rttm_path)

        original_keys = tuple(rttm.keys())

        # The default scripts have a strange convention and add some postfixes
        # that have to be removed. e.g.:
        # S02_U06.ENH or S02_U06
        rttm = {
            k.replace('_U06', '').replace('.ENH', ''): v
            for k, v in rttm.items()
        }
        assert len(rttm.keys()) == len(original_keys), (
        rttm.keys(), original_keys)

        return rttm

    @staticmethod
    def example_id(file_id, speaker_id, start, end):
        """

        Args:
            file_id:
            speaker_id:
            start:
            end:

        Returns:

        >>> RTTMDatabase.example_id('S02', '1', 100, 200)
        'S02_U06.-1-000000100_000000200'

        """
        # CHiME-6 baseline needs a special pattern for the example id
        # Cannot drop U06 for CHiME-6:
        # The kaldi baseline can only be executed, when the
        # utterance ID contains U06, independently if this array
        # was used.
        # local/get_hyp_perspeaker_perarray_file.py expects:
        #     uttid_id = 'S09_U06.ENH-4-704588-704738'
        #     micid = uttid_id.strip().split('.')[0].strip().split('_')[1]
        #     speakerid = uttid_id.strip().split('-')[1]
        #     sessionid = uttid_id.strip().split('_')[0]

        # use zfill to ensure correct sorting later
        max_digits = len(str(16000 * 60 * 60 * 10))  # 10h
        start = str(start).zfill(max_digits)
        end = str(end).zfill(max_digits)

        return f'{file_id}_U06.-{speaker_id}-{start}_{end}'

    @cached_property
    def data(self):
        datasets = {}
        for session_id, session in self._rttm.items():
            datasets[session_id] = {}
            for speaker_id, speaker in session.items():
                for start, end in speaker.intervals:
                    example_id = self.example_id(session_id, speaker_id, start, end)

                    try:
                        audio_path = self._audio_paths[session_id]
                    except KeyError as e:
                        raise ValueError(
                            'self._audio_paths does not contain the session_id'
                            f'{session_id!r},\nit has only: '
                            f'{list(self._audio_paths.keys())}'
                        )

                    datasets[session_id][example_id] = {
                        'example_id': example_id,
                        'start': start,
                        'end': end,
                        'num_samples': end - start,
                        'session_id': session_id,
                        'speaker_id': speaker_id,
                        'audio_path': audio_path,
                    }
        return {
            'datasets': datasets,
            'alias': self._alias,
        }

    def get_dataset_for_session(
            self,
            session,
            *,
            audio_read=False,
            # drop_unknown_target_speaker=False,
            adjust_times=False,
            context_samples=0,
            equal_start_context=False,
    ):
        from pb_chime5.database.chime5.database import (
            backup_orig_start_end,
            adjust_start_end,
            AddContext,
        )

        if isinstance(session, str):
            session = (session,)

        it = self.get_dataset(session)

        # # Ignore drop_unknown_target_speaker
        # if drop_unknown_target_speaker:
        #     it = it.filter(lambda ex: ex['transcription'] != '[redacted]', lazy=False)

        if context_samples is not 0 or adjust_times:
            it = it.map(backup_orig_start_end)

        if adjust_times:
            if adjust_times is True:
                # assert drop_unknown_target_speaker, (
                #     'adjust_times is undefined for '
                #     'ex["target_speaker"] == "unknown". '
                #     'Set adjust_times to True.'
                # )
                it = it.map(adjust_start_end)
            else:
                raise ValueError(adjust_times)

        if context_samples is not 0:
            # adjust_times should be before AddContext, because AddContext
            # adds the new key start_context that
            it = it.map(AddContext(
                context_samples,
                equal_start_context=equal_start_context,
            ))

        if audio_read is False:
            pass
        elif audio_read is True:

            def load_audio(example):
                min_num_samples = example.get('end_orig', example['end']) - example['start']
                example['audio_data'] = recursive_load_audio(
                    example['audio_path'],
                    start=example['start'],
                    stop=example['end'],
                    min_num_samples=min_num_samples,
                )
                return example

            it = it.map(load_audio)
        else:
            raise TypeError(audio_read)

        return it


def recursive_load_audio(
        path,
        *,
        frames=-1,
        start=0,
        stop=None,
        dtype=np.float64,
        fill_value=None,
        expected_sample_rate=None,
        unit='samples',
        return_sample_rate=False,
        min_num_samples=1,
):
    """
    Recursively loads all leafs (i.e. tuple/list entry or dict value) in the
    object `path`. `path` can be a nested structure, but can also be a str or
    pathlib.Path. When the entry type was a tuple or list, try to convert that
    object to a np.array with a dytpe different from np.object.

    Differences to paderbox.io.recursive_load_audio:
     - Ignore to short utterances (num_samples < min_num_samples)
     - Assume that shorter audio files come from different lengths of the audio
       file. Drop the samples from the longer audio files.
       This is useful for CHiME-6, in databases it may cause problems.

    For an explanation of the arguments, see `load_audio`.

    >>> from paderbox.testing.testfile_fetcher import get_file_path
    >>> from paderbox.notebook import pprint
    >>> path1 = get_file_path('speech.wav')
    >>> path2 = get_file_path('sample.wav')
    >>> pprint(recursive_load_audio(path1))
    array(shape=(49600,), dtype=float64)
    >>> pprint(recursive_load_audio([path1, path1]))
    array(shape=(2, 49600), dtype=float64)
    >>> pprint(recursive_load_audio([path1, path2]))
    [array(shape=(49600,), dtype=float64), array(shape=(38520,), dtype=float64)]
    >>> pprint(recursive_load_audio({'a': path1, 'b': path1}))
    {'a': array(shape=(49600,), dtype=float64),
     'b': array(shape=(49600,), dtype=float64)}
    >>> pprint(recursive_load_audio([path1, (path2, path2)]))
    [array(shape=(49600,), dtype=float64), array(shape=(2, 38520), dtype=float64)]

    """
    kwargs = locals().copy()
    path = kwargs.pop('path')

    if isinstance(path, (tuple, list)):
        data = [recursive_load_audio(a, **kwargs) for a in path]

        len_data = len(data)
        data = [d for d in data if d is not None]
        # Only allow 2 missing arrays
        assert len(data) >= len_data - 8, (len(data), len_data)

        np_data = np.array(data)
        if np_data.dtype != np.object:
            return np_data

        # ToDo: Support outer list, e.g. a list of dicts

        all_num_samples = [d.shape[-1] for d in data]
        num_samples = min(all_num_samples)
        assert num_samples >= min_num_samples, (num_samples, min_num_samples, all_num_samples)

        data = [d[..., :num_samples] for d in data]

        np_data = np.array(data)
        if np_data.dtype != np.object:
            return np_data

        return data
    elif isinstance(path, dict):
        return {k: recursive_load_audio(v, **kwargs) for k, v in path.items()}
    else:
        min_num_samples = kwargs.pop('min_num_samples')
        data = pb.io.load_audio(path, **kwargs)
        data: np.ndarray

        # Drop, when hitting e.g. "Last 15 minutes of U05 missing"
        if data.shape[-1] < min_num_samples:
            return None
        return data