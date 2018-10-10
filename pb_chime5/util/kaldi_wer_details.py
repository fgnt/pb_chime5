import functools
from pathlib import Path
from collections import defaultdict, Counter
import re

import colorama
from cached_property import cached_property

import pandas as pd


from nt.io.data_dir import database_jsons
from nt.database import chime5
from chime5.io.file_cache import file_cache
from chime5.scripts.create_mapping_json import Chime5KaldiToNtIdMapping


class GetNumSamples(dict):
    def __init__(self, db, kaldi_to_nt_id_mapper):
        self.db = db
        self.kaldi_to_nt_id_mapper = kaldi_to_nt_id_mapper
        super().__init__()

    @cached_property
    def it(self):
        return self.db.get_iterator_by_names('dev')
        # db.get_iterator_for_session('S02', audio_read=False, drop_unknown_target_speaker=True, adjust_times=True)

    def __missing__(self, example_id):
        ex = self.it[self.kaldi_to_nt_id_mapper[example_id]]
        _, _, array_id, _ = example_id.split('_')
        num_samples = ex['num_samples']['observation'][array_id]

        self[example_id] = num_samples
        return num_samples


class ReaderKaldiWERDetailsPerUtt:

    def __init__(
            self,
            db=None,
            num_samples=False,
            kaldi_to_nt_id_mapping_json=None
    ):
        if db is None:
            self.db = chime5.Chime5(database_jsons / 'chime5_orig.json')
        else:
            self.db = db
        self.num_samples = num_samples
        if kaldi_to_nt_id_mapping_json is None:
            self.kaldi_to_nt_id_mapper = Chime5KaldiToNtIdMapping()
        else:
            self.kaldi_to_nt_id_mapper = Chime5KaldiToNtIdMapping(kaldi_to_nt_id_mapping_json)

        if num_samples:
            self.num_sample_getter = GetNumSamples(self.db, self.kaldi_to_nt_id_mapper)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.db})'

    @functools.lru_cache()
    @file_cache('read_kaldi_wer_details_per_utt', backend='pickle', verbose=True)
    def __call__(
            self,
            p,
            as_dataframe=True,
            use_nt_id=False
    ) -> pd.DataFrame:
        """
        >>> file = '/net/vol/jensheit/kaldi/egs/chime5/inear_bss_cacgmm_v3/finetune_0/kaldi/exp/chain_worn_bss_stereo_cleaned/tdnn1a_sp/decode_bss_beam_10/scoring_kaldi/wer_details/per_utt'
        >>> df = ReaderKaldiWERDetailsPerUtt()(file)  #doctest: +ELLIPSIS
        file_cache read_kaldi_wer_details_per_utt: ... <function ReaderKaldiWERDetailsPerUtt.__call__ at 0x...>(<chime5.visualization.kaldi_wer_details.ReaderKaldiWERDetailsPerUtt object at 0x...>, /net/vol/jensheit/kaldi/egs/chime5/inear_bss_cacgmm_v3/finetune_0/kaldi/exp/chain_worn_bss_stereo_cleaned/tdnn1a_sp/decode_bss_beam_10/scoring_kaldi/wer_details/per_utt)
        >>> df.iloc[0]
        ref                     it's the blue i think i think
        len                                                 7
        hyp                       *** *** *** *** *** *** ***
        op                                      D D D D D D D
        levenshtein_distance                                7
        rel_wer                                             1
        #csid                                         0 0 0 7
        num_samples                                     51520
        Name: P05_S02_U02_KITCHEN.ENH-0004062-0004384, dtype: object
        >>> from pprint import pprint
        >>> pprint(list(ReaderKaldiWERDetailsPerUtt()(file, as_dataframe=False, use_nt_id=True).items())[0])
        file_cache read_kaldi_wer_details_per_utt: load <function ReaderKaldiWERDetailsPerUtt.__call__ at 0x7f68b4970598>(ReaderKaldiWERDetailsPerUtt(Chime5(PosixPath('/home/cbj/storage/database_jsons/chime5_orig.json'))), /net/vol/jensheit/kaldi/egs/chime5/inear_bss_cacgmm_v3/finetune_0/kaldi/exp/chain_worn_bss_stereo_cleaned/tdnn1a_sp/decode_bss_beam_10/scoring_kaldi/wer_details/per_utt, as_dataframe=False, use_nt_id=True)
        ('P05_S02_0004060-0004382',
         {'#csid': '0 0 0 7',
          'cor': 0,
          'del': 7,
          'hyp': '*** *** *** *** *** *** ***',
          'ins': 0,
          'len': 7,
          'levenshtein_distance': 7,
          'num_samples': 51520,
          'op': 'D D D D D D D',
          'ref': "it's the blue i think i think",
          'rel_wer': 1.0,
          'sub': 0})

        """
        p = Path(p).expanduser().resolve()
        if not p.exists():
            for parent_path in p.parents:
                if parent_path.exists():
                    print(
                        f'{colorama.Fore.GREEN}{parent_path}{colorama.Fore.RESET} exists but {colorama.Fore.RED}{p.relative_to(parent_path)}{colorama.Fore.RESET} does not exists.')
                    print(f'The subfolders in {parent_path} are')
                    for tmp in sorted(parent_path.glob('*')):
                        print(f'    {tmp.relative_to(parent_path)}')
                    assert p.exists(), p

        if p.name != 'per_utt':
            p = p / 'scoring_kaldi/wer_details/per_utt'

        assert p.is_file(), p

        text = p.read_text()

        r = re.compile(r'P\d\d_S\d\d_(?P<array_id>U\d\d)_')

        d = defaultdict(dict)
        for line in text.splitlines():
            example_id, line_type, *content = line.split()

            if example_id in d:
                assert line_type not in d[example_id], (example_id, line_type)

            for c in content:
                assert ' ' not in c, content

            d[example_id][line_type] = ' '.join(content)

            if line_type == 'ref':
                d[example_id]['len'] = len([c for c in content if c != '***'])

                # nt_example_id = kaldi_to_nt_id_mapper[example_id]
                # array_id = r.match('P05_S02_U02_KITCHEN.ENH-0004062-0004384').group(
                #     "array_id")
                # d[example_id]['overlap'] = overlap[nt_example_id][array_id]
                # d[example_id]['overlap_non_sil_cross'] = \
                # overlap_non_sil_cross[nt_example_id][array_id]

            if line_type == 'op':
                c = Counter(content)
                d[example_id]['len2'] = c['C'] + c['S'] + c['D']
                d[example_id]['levenshtein_distance'] = c['S'] + c['D'] + c['I']

                d[example_id]['del'] = c['D']
                d[example_id]['ins'] = c['I']
                d[example_id]['sub'] = c['S']
                d[example_id]['cor'] = c['C']

                #         print(example_id)
                if d[example_id]['len2'] > 0:
                    d[example_id]['rel_wer'] = d[example_id][
                                                   'levenshtein_distance'] / \
                                               d[example_id]['len2']
                else:
                    if d[example_id]['levenshtein_distance'] > 0:
                        d[example_id]['rel_wer'] = -1.
                    else:
                        d[example_id]['rel_wer'] = 0.

            if 'len' in d[example_id] and 'len2' in d[example_id]:
                assert d[example_id]['len'] == d[example_id]['len2'], (
                example_id, d[example_id]['len'], d[example_id]['len2'], c)
                del d[example_id]['len2']

        if self.num_samples:
            for example_id in d.keys():
                d[example_id]['num_samples'] = self.num_sample_getter[example_id]

        if use_nt_id:
            d_tmp = d
            d = {self.kaldi_to_nt_id_mapper[k]: v for k, v in d.items()}
            assert len(d_tmp) == len(d), (len(d), len(d_tmp))

        if as_dataframe:
            df = pd.DataFrame.from_dict(d, orient='index')
            del df['del']
            del df['cor']
            del df['sub']
            del df['ins']

            #     print(df.dtypes)
            #     df = pd.DataFrame(df.to_dict())  # recomute the column types
            #     print(df.dtypes)

            return df
        else:
            return d


    # df_tmp = read_kaldi_wer_details_per_utt(
    #     '/net/vol/jensheit/kaldi/egs/chime5/inear_bss_cacgmm_v3/finetune_0/kaldi/exp/chain_worn_bss_stereo_cleaned/tdnn1a_sp/decode_bss_beam_10/scoring_kaldi/wer_details/per_utt')
    # df_tmp.head()