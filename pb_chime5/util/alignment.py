from pathlib import Path
import collections
import functools
import itertools
import operator

import numpy as np

from nt.utils.mapping import Dispatcher
from nt.database.chime5 import kaldi_to_nt_example_id, kaldi_id_to_channel, activity_frequency_to_time

from pb_chime5.util.alignment_util import cy_alignment_id2phone


def get_phone_alignment(
        ali_path,
        use_kaldi_id=False,
        unique_per_utt=True,
        channel_preference=None,
):
    """

    use_kaldi_id:
        Use a unique id per utterance or tha kaldi id (i.e. array dependent)
    unique_per_utt:
        Return one per utterance. When multiple kaldi ids are available use
        channel_preference.
    channel_preference:
        None or list of channels.
        Example channel_preference = ['R', 'L']
         - assert any alignment has a left channel and any alignment has a right
           channel. (Note any not all)
         - If an example has a left and right channel, select the right.

    >>> # np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')
    >>> np.set_printoptions(threshold=50, edgeitems=30)
    >>> from IPython.lib.pretty import pprint
    >>> p = Path('/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali')
    >>> # p = ('~/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali', '~/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali')
    >>> alignment = get_phone_alignment(p)
    >>> pprint(alignment['P06_S02_0060700-0061058'])  # doctest: +ELLIPSIS
    array(['sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'd_B', 'd_B', 'd_B', 'd_B', 'd_B', 'd_B', 'ih_I', 'ih_I', 'ih_I',
           'z_E', 'z_E', 'z_E', ..., 'ay_I', 'ay_I', 'ay_I', 't_E', 't_E',
           't_E', 't_E', 't_E', 't_E', 't_E', 't_E', 't_E', 't_E', 't_E',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil'], dtype='<U4')
    >>> pprint(alignment['P25_S09_0121800-0122035'])  # doctest: +ELLIPSIS
    array(['sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'ay_B', 'ay_B', 'ay_B', 'ay_B', 'ay_B', 'm_E', 'm_E', 'm_E',
           'g_B', 'g_B', 'g_B', 'aa_I', 'aa_I', 'aa_I', 'aa_I', 'n_I', 'n_I',
           'n_I', 'ah_E', 'ah_E', ..., 'n_E', 'n_E', 'n_E', 'n_E', 'n_E',
           'n_E', 'n_E', 'n_E', 'n_E', 'n_E', 'n_E', 'n_E', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil'], dtype='<U4')
    >>> non_sil_alignment = {k: v != 'sil' for k, v in alignment.items()}
    >>> pprint(dict(list(non_sil_alignment.items())[:3]))  # doctest: +ELLIPSIS
    {'P05_S02_0004060-0004382': array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True, False, False, ...,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False]),
     'P05_S02_0007011-0007297': array([False, False, False, False, False,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True, ..., False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False]),
     'P05_S02_0007437-0007908': array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
             True,  True,  True, ...,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False])}

    # >>> p = '/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_cleaned_ali_train_worn_u100k_cleaned_sp'
    # >>> alignment = get_phone_alignment(p)
    # >>> pprint(dict(list(non_sil_alignment.items())[:3]))  # doctest: +ELLIPSIS
    # >>> print(len(alignment))

    >>> ali_path = (
    ...     '/net/vol/jensheit/kaldi/egs/chime5/inear_bss_cacgmm_v3/finetune_0/kaldi/exp/tri3_worn_bss_stereo_train_worn_bss_stereo_ali/',
    ...     '/net/vol/jensheit/kaldi/egs/chime5/inear_bss_cacgmm_v3/finetune_0/kaldi/exp/tri3_worn_bss_stereo_dev_worn_bss_stereo_ali/',
    ... )  # slow because of train
    >>> alignment = get_phone_alignment(ali_path, channel_preference=['R', 'L'])
    >>> pprint(alignment['P06_S02_0060700-0061058'])  # doctest: +ELLIPSIS
    array(['sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'd_B', 'd_B', 'd_B', 'd_B', 'ih_I', 'ih_I',
           'ih_I', 'z_E', 'z_E', ..., 't_E', 't_E', 't_E', 't_E', 't_E',
           't_E', 't_E', 't_E', 't_E', 't_E', 't_E', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil',
           'sil', 'sil', 'sil', 'sil', 'sil', 'sil', 'sil'], dtype='<U4')


    """
    import nt.kaldi

    if isinstance(ali_path, (tuple, list)):
        alignments_list = [
            get_phone_alignment(
                ap,
                channel_preference=channel_preference,
                use_kaldi_id=use_kaldi_id,
            )
            for ap in ali_path
        ]
        total_len = sum([len(a) for a in alignments_list])
        alignments = {
            k: v
            for a in alignments_list
            for k, v in a.items()
        }
        assert len(alignments) == total_len
        return alignments

    ali_path = Path(ali_path).expanduser().resolve()

    tmp = [reversed(line.split()) for line in
           (ali_path / 'phones.txt').read_text().splitlines()]
    id2phone = {int(k): v for k, v in tmp}
    assert len(id2phone) == len(tmp)

    _alignments = nt.kaldi.alignment.import_alignment_data(
        ali_path,
        import_fn=nt.kaldi.alignment.import_phone_alignment_from_file,
        per_frame=True,
        model_name=ali_path / 'final.mdl'
    )

    alignments = _helper(
        _alignments,
        channel_preference=channel_preference,
        # id2phone=id2phone,
        unique_per_utt=unique_per_utt,
        use_kaldi_id=use_kaldi_id,
    )

    return Dispatcher(cy_alignment_id2phone(alignments, id2phone))


def _helper(
        _alignments,
        channel_preference,
        unique_per_utt=True,
        use_kaldi_id=False,
):
    """
    >>> from IPython.lib.pretty import pprint
    >>> alignments = {
    ...     'P28_S09_LIVING.R-0714562-0714764': [1],
    ...     'P28_S09_LIVING.L-0714562-0714764': [2],
    ...     'P09_S03_NOLOCATION.L-0007974-0008116': [3],
    ...     'P09_S03_NOLOCATION.R-0008255-0008300': [4]
    ...     # 'P05_S02_U02_KITCHEN.ENH-0007012-0007298': [3],
    ...     # 'P09_S03_U01_NOLOCATION.CH1-0005948-0006038': [4],
    ... }
    >>> pprint(_helper(alignments, channel_preference=['R', 'L']))
    {'P09_S03_0007974-0008116': [3],
     'P09_S03_0008255-0008300': [4],
     'P28_S09_0714562-0714764': [1]}
    >>> pprint(_helper(alignments, channel_preference=['L', 'R']))
    {'P09_S03_0007974-0008116': [3],
     'P09_S03_0008255-0008300': [4],
     'P28_S09_0714562-0714764': [2]}
    >>> pprint(_helper(alignments, channel_preference=['L', 'R'], use_kaldi_id=True))
    {'P09_S03_NOLOCATION.L-0007974-0008116': [3],
     'P09_S03_NOLOCATION.R-0008255-0008300': [4],
     'P28_S09_LIVING.L-0714562-0714764': [2]}
    >>> pprint(_helper(alignments, channel_preference=['L']))
    Traceback (most recent call last):
    ...
    AssertionError: Expect channels ['L'] but found ('L', 'R').
    >>> pprint(_helper(alignments, channel_preference=None, use_kaldi_id=True))
    Traceback (most recent call last):
    ...
    AssertionError: Item for key 'P28_S09_0714562-0714764' is not unique.
    In the 0. dict is the value ('P28_S09_LIVING.L-0714562-0714764', [2]), while in the merged ('P28_S09_LIVING.R-0714562-0714764', [1]).
    Intersection keys: {'P28_S09_0714562-0714764'}
    >>> pprint(_helper(alignments, channel_preference=None, use_kaldi_id=True, unique_per_utt=False))
    {'P09_S03_NOLOCATION.L-0007974-0008116': [3],
     'P09_S03_NOLOCATION.R-0008255-0008300': [4],
     'P28_S09_LIVING.L-0714562-0714764': [2],
     'P28_S09_LIVING.R-0714562-0714764': [1]}

    >>> alignments = {
    ...     'P28_S09_LIVING.L-0714562-0714764': [2],
    ...     'P09_S03_NOLOCATION.L-0007974-0008116': [3],
    ... }
    >>> pprint(_helper(alignments, ('R', 'L')))
    Traceback (most recent call last):
    ...
    AssertionError: Expect channels ('R', 'L') but found ('L',).
    >>> alignments = {
    ...     'P28_S09_LIVING.R-0714562-0714764': [1],
    ...     # 'P28_S09_LIVING.L-0714562-0714764': [2],
    ...     'P09_S03_NOLOCATION.L-0007974-0008116': [3],
    ...     'P09_S03_NOLOCATION.R-0008255-0008300': [4]
    ...     # 'P05_S02_U02_KITCHEN.ENH-0007012-0007298': [3],
    ...     # 'P09_S03_U01_NOLOCATION.CH1-0005948-0006038': [4],
    ... }
    >>> pprint(_helper(alignments, channel_preference=None, use_kaldi_id=True))
    {'P09_S03_NOLOCATION.L-0007974-0008116': [3],
     'P09_S03_NOLOCATION.R-0008255-0008300': [4],
     'P28_S09_LIVING.R-0714562-0714764': [1]}

    """

    alignments = [
        (
            kaldi_id_to_channel(k),
            # kaldi_to_nt_example_id does not work for kaldi array ids
            kaldi_to_nt_example_id(k) if not use_kaldi_id else k,
            k,
            v,
        )
        for k, v in _alignments.items()
    ]

    mapping_channels_examples_data = {
        key: {
            nt_id if unique_per_utt else kaldi_id: (kaldi_id, data)
            for channel, nt_id, kaldi_id, data in sub_iterator
        }
        for key, sub_iterator in itertools.groupby(
            sorted(alignments),
            key=lambda e: e[0],
        )
    }

    if not unique_per_utt:
        assert channel_preference is None, channel_preference
        assert use_kaldi_id is True, use_kaldi_id

    if channel_preference is None:
        from cbj.dict_utils import merge
        ret = dict(sorted(merge(*mapping_channels_examples_data.values()).items()))
    else:
        assert len(mapping_channels_examples_data) == len(channel_preference), f'Expect channels {channel_preference} but found {tuple(mapping_channels_examples_data.keys())}.'

        # Get values with order of channel_preference
        sorted_mapping_channels_examples_data = \
            operator.itemgetter(*channel_preference)(mapping_channels_examples_data)

        ret = dict(sorted(collections.ChainMap(*sorted_mapping_channels_examples_data).items()))

    if use_kaldi_id:
        return {
            kaldi_id: ali
            for nt_id, (kaldi_id, ali) in ret.items()
        }
    else:
        return {
            nt_id: ali
            for nt_id, (kaldi_id, ali) in ret.items()
        }
