from collections import defaultdict

from nt.utils.mapping import Dispatcher
from nt.database.chime5 import Chime5


session_speakers_mapping = Dispatcher({
    'S02': ['P05', 'P06', 'P07', 'P08'],
    'S03': ['P09', 'P10', 'P11', 'P12'],
    'S04': ['P09', 'P10', 'P11', 'P12'],
    'S05': ['P13', 'P14', 'P15', 'P16'],
    'S06': ['P13', 'P14', 'P15', 'P16'],
    'S07': ['P17', 'P18', 'P19', 'P20'],
    'S08': ['P21', 'P22', 'P23', 'P24'],
    'S09': ['P25', 'P26', 'P27', 'P28'],
    'S12': ['P33', 'P34', 'P35', 'P36'],
    'S13': ['P33', 'P34', 'P35', 'P36'],
    'S16': ['P21', 'P22', 'P23', 'P24'],
    'S17': ['P17', 'P18', 'P19', 'P20'],
    'S18': ['P41', 'P42', 'P43', 'P44'],
    'S19': ['P49', 'P50', 'P51', 'P52'],
    'S20': ['P49', 'P50', 'P51', 'P52'],
    'S22': ['P41', 'P42', 'P43', 'P44'],
    'S23': ['P53', 'P54', 'P55', 'P56'],
    'S24': ['P53', 'P54', 'P55', 'P56'],
})


session_dataset_mapping = Dispatcher({'S02': 'dev',
    'S03': 'train',
    'S04': 'train',
    'S05': 'train',
    'S06': 'train',
    'S07': 'train',
    'S08': 'train',
    'S09': 'dev',
    'S12': 'train',
    'S13': 'train',
    'S16': 'train',
    'S17': 'train',
    'S18': 'train',
    'S19': 'train',
    'S20': 'train',
    'S22': 'train',
    'S23': 'train',
    'S24': 'train'
})

session_array_to_num_samples_mapping = Dispatcher({
    'S02_P': 142464640,
    'S02_U01': 142464640,
    'S02_U02': 142464640,
    'S02_U03': 142464640,
    'S02_U04': 142464640,
    'S02_U05': 142464640,
    'S02_U06': 142464640,
    'S03_P': 126116800,
    'S03_U01': 126116800,
    'S03_U02': 126116800,
    'S03_U03': 126116800,
    'S03_U04': 126116800,
    'S03_U05': 126116800,
    'S03_U06': 126116800,
    'S04_P': 143615200,
    'S04_U01': 143615200,
    'S04_U02': 143615200,
    'S04_U03': 143615200,
    'S04_U04': 143615200,
    'S04_U05': 143615200,
    'S04_U06': 143615200,
    'S05_P': 145660480,
    'S05_U01': 145660480,
    'S05_U02': 145660480,
    'S05_U04': 144414928,
    'S05_U05': 145660480,
    'S05_U06': 145660480,
    'S06_P': 144103520,
    'S06_U01': 144103520,
    'S06_U02': 144103520,
    'S06_U03': 144103520,
    'S06_U04': 144103520,
    'S06_U05': 144103520,
    'S06_U06': 144103520,
    'S07_P': 141009280,
    'S07_U01': 141009280,
    'S07_U02': 141009280,
    'S07_U03': 141009280,
    'S07_U04': 141009280,
    'S07_U05': 141009280,
    'S07_U06': 141009280,
    'S08_P': 145521280,
    'S08_U01': 145521280,
    'S08_U02': 145521280,
    'S08_U03': 145521280,
    'S08_U04': 145521280,
    'S08_U05': 145521280,
    'S08_U06': 145521280,
    'S09_P': 114583520,
    'S09_U01': 114583520,
    'S09_U02': 114583520,
    'S09_U03': 114583520,
    'S09_U04': 114583520,
    'S09_U06': 114583520,
    'S12_P': 143421600,
    'S12_U01': 143421600,
    'S12_U02': 143421600,
    'S12_U03': 143421600,
    'S12_U04': 143421600,
    'S12_U05': 129600000,
    'S12_U06': 143421600,
    'S13_P': 144170880,
    'S13_U01': 144170880,
    'S13_U02': 144170880,
    'S13_U03': 144170880,
    'S13_U04': 144170880,
    'S13_U05': 144170880,
    'S13_U06': 144170880,
    'S16_P': 146221440,
    'S16_U01': 146221440,
    'S16_U02': 146221440,
    'S16_U03': 146221440,
    'S16_U04': 146221440,
    'S16_U05': 146221440,
    'S16_U06': 146221440,
    'S17_P': 146177280,
    'S17_U01': 146177280,
    'S17_U02': 146177280,
    'S17_U03': 146177280,
    'S17_U04': 146177280,
    'S17_U05': 146177280,
    'S17_U06': 146177280,
    'S18_P': 155882080,
    'S18_U01': 155882080,
    'S18_U02': 155882080,
    'S18_U03': 155882080,
    'S18_U04': 155882080,
    'S18_U05': 155882080,
    'S18_U06': 155592867,
    'S19_P': 146522240,
    'S19_U01': 146522240,
    'S19_U02': 146522240,
    'S19_U03': 146522240,
    'S19_U04': 146522240,
    'S19_U05': 146522240,
    'S19_U06': 146522240,
    'S20_P': 132542400,
    'S20_U01': 132542400,
    'S20_U02': 132542400,
    'S20_U03': 132542400,
    'S20_U04': 132542400,
    'S20_U05': 132542400,
    'S20_U06': 132542400,
    'S22_P': 149503360,
    'S22_U01': 149503360,
    'S22_U02': 149503360,
    'S22_U04': 149503360,
    'S22_U05': 149503360,
    'S22_U06': 149503360,
    'S23_P': 171567520,
    'S23_U01': 171567520,
    'S23_U02': 171567520,
    'S23_U03': 171567520,
    'S23_U04': 171567520,
    'S23_U05': 171567520,
    'S23_U06': 171567520,
    'S24_P': 150862080,
    'S24_U01': 150862080,
    'S24_U02': 150862080,
    'S24_U03': 150862080,
    'S24_U04': 150862080,
    'S24_U05': 150862080,
    'S24_U06': 150862080,
})


def _get_session_speaker_mapping():
    """
    Helper for testcase and documentation.

    >>> from IPython.lib.pretty import pprint
    >>> session_speaker_mapping_calc = _get_session_speaker_mapping()
    >>> pprint(session_speaker_mapping_calc)
    {'S02': ['P05', 'P06', 'P07', 'P08'],
     'S03': ['P09', 'P10', 'P11', 'P12'],
     'S04': ['P09', 'P10', 'P11', 'P12'],
     'S05': ['P13', 'P14', 'P15', 'P16'],
     'S06': ['P13', 'P14', 'P15', 'P16'],
     'S07': ['P17', 'P18', 'P19', 'P20'],
     'S08': ['P21', 'P22', 'P23', 'P24'],
     'S09': ['P25', 'P26', 'P27', 'P28'],
     'S12': ['P33', 'P34', 'P35', 'P36'],
     'S13': ['P33', 'P34', 'P35', 'P36'],
     'S16': ['P21', 'P22', 'P23', 'P24'],
     'S17': ['P17', 'P18', 'P19', 'P20'],
     'S18': ['P41', 'P42', 'P43', 'P44'],
     'S19': ['P49', 'P50', 'P51', 'P52'],
     'S20': ['P49', 'P50', 'P51', 'P52'],
     'S22': ['P41', 'P42', 'P43', 'P44'],
     'S23': ['P53', 'P54', 'P55', 'P56'],
     'S24': ['P53', 'P54', 'P55', 'P56']}
    >>> assert session_speaker_mapping_calc == session_speaker_mapping

    """
    db = Chime5()
    it_dev = db.get_iterator_by_names('dev')
    it_train = db.get_iterator_by_names('train')
    summary = defaultdict(set)
    # ex = it_dev[0]
    for ex in it_dev:
        summary[ex['session_id']] |= set(ex['speaker_id'])
    for ex in it_train:
        summary[ex['session_id']] |= set(ex['speaker_id'])
    return dict(sorted(map((lambda item: (item[0], list(sorted(item[1])))), summary.items())))


def _get_session_dataset_mapping():
    """


    >>> from IPython.lib.pretty import pprint
    >>> session_dataset_mapping_calc = _get_session_dataset_mapping()
    >>> pprint(session_dataset_mapping_calc)
    {'S02': 'dev',
     'S03': 'train',
     'S04': 'train',
     'S05': 'train',
     'S06': 'train',
     'S07': 'train',
     'S08': 'train',
     'S09': 'dev',
     'S12': 'train',
     'S13': 'train',
     'S16': 'train',
     'S17': 'train',
     'S18': 'train',
     'S19': 'train',
     'S20': 'train',
     'S22': 'train',
     'S23': 'train',
     'S24': 'train'}
    >>> assert session_dataset_mapping_calc == session_dataset_mapping

    """
    db = Chime5()
    summary = {}
    for dataset in ['dev', 'train']:
        it = db.get_iterator_by_names(dataset)
        for ex in it:
            assert dataset == summary.get(ex['session_id'], dataset), (
                dataset, summary.get(ex['session_id'], dataset)
            )
            summary[ex['session_id']] = dataset

    return dict(sorted(summary.items()))


def _get_session_array_to_num_samples_mapping():
    """
    Inear has always the same number of samples, but the arrays have sometimes
    some samples lost.

    >>> from IPython.lib.pretty import pprint
    >>> session_num_samples_mapping_calc = _get_session_array_to_num_samples_mapping()
    >>> pprint(session_num_samples_mapping_calc)
    {'S02_P': 142464640,
     'S02_U01': 142464640,
     'S02_U02': 142464640,
     'S02_U03': 142464640,
     'S02_U04': 142464640,
     'S02_U05': 142464640,
     'S02_U06': 142464640,
     'S03_P': 126116800,
     'S03_U01': 126116800,
     'S03_U02': 126116800,
     'S03_U03': 126116800,
     'S03_U04': 126116800,
     'S03_U05': 126116800,
     'S03_U06': 126116800,
     'S04_P': 143615200,
     'S04_U01': 143615200,
     'S04_U02': 143615200,
     'S04_U03': 143615200,
     'S04_U04': 143615200,
     'S04_U05': 143615200,
     'S04_U06': 143615200,
     'S05_P': 145660480,
     'S05_U01': 145660480,
     'S05_U02': 145660480,
     'S05_U04': 144414928,
     'S05_U05': 145660480,
     'S05_U06': 145660480,
     'S06_P': 144103520,
     'S06_U01': 144103520,
     'S06_U02': 144103520,
     'S06_U03': 144103520,
     'S06_U04': 144103520,
     'S06_U05': 144103520,
     'S06_U06': 144103520,
     'S07_P': 141009280,
     'S07_U01': 141009280,
     'S07_U02': 141009280,
     'S07_U03': 141009280,
     'S07_U04': 141009280,
     'S07_U05': 141009280,
     'S07_U06': 141009280,
     'S08_P': 145521280,
     'S08_U01': 145521280,
     'S08_U02': 145521280,
     'S08_U03': 145521280,
     'S08_U04': 145521280,
     'S08_U05': 145521280,
     'S08_U06': 145521280,
     'S09_P': 114583520,
     'S09_U01': 114583520,
     'S09_U02': 114583520,
     'S09_U03': 114583520,
     'S09_U04': 114583520,
     'S09_U06': 114583520,
     'S12_P': 143421600,
     'S12_U01': 143421600,
     'S12_U02': 143421600,
     'S12_U03': 143421600,
     'S12_U04': 143421600,
     'S12_U05': 129600000,
     'S12_U06': 143421600,
     'S13_P': 144170880,
     'S13_U01': 144170880,
     'S13_U02': 144170880,
     'S13_U03': 144170880,
     'S13_U04': 144170880,
     'S13_U05': 144170880,
     'S13_U06': 144170880,
     'S16_P': 146221440,
     'S16_U01': 146221440,
     'S16_U02': 146221440,
     'S16_U03': 146221440,
     'S16_U04': 146221440,
     'S16_U05': 146221440,
     'S16_U06': 146221440,
     'S17_P': 146177280,
     'S17_U01': 146177280,
     'S17_U02': 146177280,
     'S17_U03': 146177280,
     'S17_U04': 146177280,
     'S17_U05': 146177280,
     'S17_U06': 146177280,
     'S18_P': 155882080,
     'S18_U01': 155882080,
     'S18_U02': 155882080,
     'S18_U03': 155882080,
     'S18_U04': 155882080,
     'S18_U05': 155882080,
     'S18_U06': 155592867,
     'S19_P': 146522240,
     'S19_U01': 146522240,
     'S19_U02': 146522240,
     'S19_U03': 146522240,
     'S19_U04': 146522240,
     'S19_U05': 146522240,
     'S19_U06': 146522240,
     'S20_P': 132542400,
     'S20_U01': 132542400,
     'S20_U02': 132542400,
     'S20_U03': 132542400,
     'S20_U04': 132542400,
     'S20_U05': 132542400,
     'S20_U06': 132542400,
     'S22_P': 149503360,
     'S22_U01': 149503360,
     'S22_U02': 149503360,
     'S22_U04': 149503360,
     'S22_U05': 149503360,
     'S22_U06': 149503360,
     'S23_P': 171567520,
     'S23_U01': 171567520,
     'S23_U02': 171567520,
     'S23_U03': 171567520,
     'S23_U04': 171567520,
     'S23_U05': 171567520,
     'S23_U06': 171567520,
     'S24_P': 150862080,
     'S24_U01': 150862080,
     'S24_U02': 150862080,
     'S24_U03': 150862080,
     'S24_U04': 150862080,
     'S24_U05': 150862080,
     'S24_U06': 150862080}
    >>> assert session_num_samples_mapping_calc == session_array_to_num_samples_mapping

    """
    from nt.io.audioread import audio_length

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def get_audio_length(file):
        return audio_length(file)

    db = Chime5()
    summary = {}
    for dataset in ['dev', 'train']:
        it = db.get_iterator_by_names(dataset)
        for ex in it:
            for array_id, files in ex['audio_path']['observation'].items():
                length = {get_audio_length(file) for file in files}
                assert len(length) == 1, (length, files)
                length, = list(length)

                session_id = ex['session_id']

                key = f'{session_id}_{array_id}'

                length_ = summary.get(key, length)
                assert length_ == length, (length_, length_, key)

                summary[f'{session_id}_{array_id}'] = length

            for array_id, file in ex['audio_path']['worn_microphone'].items():
                if len(file) == 1:
                    # remove this, when jenkins has build the new json
                    file, = file

                length = get_audio_length(file)

                session_id = ex['session_id']

                key = f'{session_id}_{array_id}'
                key = f'{session_id}_P'

                length_ = summary.get(key, length)
                assert length_ == length, (length_, length_, key)

                summary[key] = length

    return dict(sorted(summary.items()))
