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
