"""

Legend:
n, N ... time
t, T ... frame
f, F ... frequency
d, D ... channel
a, A ... array
"""

import functools
from dataclasses import dataclass
from pathlib import Path

from cached_property import cached_property

import numpy as np

import dlp_mpi
from dlp_mpi.util import ensure_single_thread_numeric

import nara_wpe
import nara_wpe.wpe
from pb_bss.distribution import (
    CACGMMTrainer
)
from pb_bss.distribution.gammacacgmm import NoisyGammaCACGMMTrainer
from pb_bss.distribution.utils import (
    stack_parameters,
)

from pb_chime5 import git_root
from pb_chime5.utils.numpy_utils import morph
from pb_chime5.database.chime5 import activity_time_to_frequency

from pb_chime5.io import load_audio, dump_audio
from pb_chime5 import mapping

JSON_PATH = git_root / 'cache'


@dataclass
class WPE:
    taps: int
    delay: int
    iterations: int
    psd_context: int

    def __call__(self, Obs, stack=None, debug=False):
        
        if Obs.ndim == 3:
            assert stack is None, stack
            Obs = nara_wpe.wpe.wpe_v8(
                Obs.transpose(2, 0, 1),
                taps=self.taps,
                delay=self.delay,
                iterations=self.iterations,
                psd_context=self.psd_context,
            ).transpose(1, 2, 0)
        elif Obs.ndim == 4:
            if stack is True:
                _A = Obs.shape[0]
                Obs = morph('ACTF->A*CTF', Obs)
                Obs = nara_wpe.wpe.wpe_v8(
                    Obs.transpose(2, 0, 1),
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                ).transpose(1, 2, 0)
                Obs = morph('A*CTF->ACTF', Obs, A=_A)
            elif stack is False:
                Obs = nara_wpe.wpe.wpe_v8(
                    Obs.transpose(0, 3, 1, 2),
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                ).transpose(0, 2, 3, 1)
                
            else:
                raise NotImplementedError(stack)
        else:
            raise NotImplementedError(Obs.shape)

        if debug:
            self.locals = locals()

        return Obs


@dataclass  # (hash=True)
class Activity:
    type: str = 'annotation'  # ['annotation', 'path', 'rttm']
    garbage_class: bool = False
    # ali_path='/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali',
    # activity_array_ali_path='~/net/storage/jheymann/__share/jensheit/chime5/kaldi/arrayBSS_v5/exp/tri3_u_bss_js_cleaned_dev_new_bss_beam_39_ali',
    database_path: str = str(JSON_PATH / 'chime6.json')
    path: str = None

    @cached_property
    def db(self):
        if 'rttm' in str(self.database_path):  # not optimal, but should work
            from pb_chime5.database.chime5.rttm import Chime6RTTMDatabase
            from padercontrib.io.data_dir import chime_6
            return Chime6RTTMDatabase(self.database_path, chime6_dir=chime_6 / 'CHiME6')
        else:
            from pb_chime5.database.chime5 import Chime5
            return Chime5(self.database_path)

    @staticmethod
    @functools.lru_cache(1)
    def _getitem(
            session_id,
            type,
            db,
            garbage_class,
    ):
        from pb_chime5.activity import get_activity_chime6

        assert type in ['annotation'], type

        return get_activity_chime6(
            iterator=db.get_datasets(session_id),
            garbage_class=garbage_class,
            dtype=np.bool,
            non_sil_alignment_fn=None,
            debug=False,
            use_ArrayIntervall=True,
        )[session_id]

    def __getitem__(self, session_id):
        if self.type in ['annotation']:
            return self._getitem(
                session_id,
                type=self.type,
                db=self.db,
                garbage_class=self.garbage_class,
            )
        elif self.type == 'path':
            import pickle

            with open(Path(self.path) / f'{session_id}.pkl', 'rb') as fd:
                return pickle.load(fd)
        elif self.type == 'rttm':
            # todo: garbage class
            from paderbox.array import intervall as array_intervall
            data = array_intervall.from_rttm(self.path)

            original_keys = tuple(data.keys())
            # The default scripts have a strange convention and add some postfixes
            # that have to be removed. e.g.:
            # S02_U06.ENH or S02_U06

            data = {
                k.replace('_U06', '').replace('.ENH', ''): v
                for k, v in data.items()
            }
            assert len(data.keys()) == len(original_keys), (data.keys(), original_keys)

            data = data[session_id]

            if self.garbage_class is False:
                data['Noise'] = array_intervall.zeros()
            elif self.garbage_class is True:
                v: array_intervall.ArrayIntervall

                # length: This value should be large enough. Use 5 hours.
                # max([v.intervals[-1][-1] for v in data.values()])
                # length = 5 * 60 * 60 * 16000
                #        hours minutes seconds samples

                data['Noise'] = array_intervall.ones()
                # data['Noise'][:] = 1
            elif self.garbage_class is None:
                pass
            else:
                raise ValueError(self.garbage_class)
        else:
            raise ValueError(type)
        return data


@dataclass
class GSS:
    iterations: int
    iterations_post: int

    verbose: bool = True

    # use_pinv: bool = False
    # stable: bool = True

    mm: str = 'cACGMM'

    def __call__(self, Obs, acitivity_freq, debug=False):

        initialization = np.asarray(acitivity_freq, dtype=np.float64)
        initialization = np.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / np.sum(initialization, keepdims=True,
                                                axis=0)
        initialization = np.repeat(initialization[None, ...], 513, axis=0)

        source_active_mask = np.asarray(acitivity_freq, dtype=np.bool)
        source_active_mask = np.repeat(source_active_mask[None, ...], 513, axis=0)

        if self.mm == 'cACGMM':
            cacGMM = CACGMMTrainer()
        elif self.mm == 'NoisyGammacACGMM':
            cacGMM = NoisyGammaCACGMMTrainer()
        else:
            raise ValueError(self.mm)

        if debug:
            learned = []
        all_affiliations = []
        F = Obs.shape[-1]
        T = Obs.T.shape[-2]
        for f in range(F):
            if self.verbose:
                if f % 50 == 0:
                    print(f'{f}/{F}')

            # T: Consider end of signal.
            # This should not be nessesary, but activity is for inear and not for
            # array.
            cur = cacGMM.fit(
                y=Obs.T[f, ...],
                initialization=initialization[f, ..., :T],
                iterations=self.iterations,
                source_activity_mask=source_active_mask[f, ..., :T],
                # return_affiliation=True,
            )

            if self.iterations_post != 0:
                if self.iterations_post != 1:
                    cur = cacGMM.fit(
                        y=Obs.T[f, ...],
                        initialization=cur,
                        iterations=self.iterations_post - 1,
                    )
                affiliation = cur.predict(
                    Obs.T[f, ...],
                )
            else:
               affiliation = cur.predict(
                   Obs.T[f, ...],
                   source_activity_mask=source_active_mask[f, ..., :T]
               )

            if debug:
                learned.append(cur)
            all_affiliations.append(affiliation)

        posterior = np.array(all_affiliations).transpose(1, 2, 0)

        if debug:
            learned = stack_parameters(learned)
            self.locals = locals()

        return posterior


def start_end_context_frames(ex, stft_size, stft_shift, stft_fading):
    start_context_samples = ex['start_orig'] - ex['start']
    end_context_samples = ex['end'] - ex['end_orig']

    assert start_context_samples >= 0, (start_context_samples, ex)
    assert end_context_samples >= 0, (end_context_samples, ex)

    from nara_wpe.utils import _samples_to_stft_frames

    start_context_frames = _samples_to_stft_frames(
        start_context_samples,
        size=stft_size,
        shift=stft_shift,
        fading=stft_fading,
    )
    end_context_frames = _samples_to_stft_frames(
        end_context_samples,
        size=stft_size,
        shift=stft_shift,
        fading=stft_fading,
    )
    return start_context_frames, end_context_frames


@dataclass
class Beamformer:
    type: str
    postfilter: str

    def __call__(self, Obs, target_mask, distortion_mask, debug=False):
        bf = self.type

        if bf == 'mvdrSouden_ban':
            from pb_chime5.speech_enhancement.beamforming_wrapper import (
                beamform_mvdr_souden_from_masks
            )
            X_hat = beamform_mvdr_souden_from_masks(
                Y=Obs,
                X_mask=target_mask,
                N_mask=distortion_mask,
                ban=True,
            )
        elif bf == 'ch2':
            X_hat = Obs[2]
        elif bf == 'sum':
            X_hat = np.sum(Obs, axis=0)
        # elif bf is None:
        #     X_hat = Obs
        else:
            raise NotImplementedError(bf)

        if self.postfilter is None:
            pass
        elif self.postfilter == 'mask_mul':
            X_hat = X_hat * target_mask
        else:
            raise NotImplementedError(self.postfilter)

        if debug:
            self.locals = locals()

        return X_hat


@dataclass
class Enhancer:
    wpe_block: WPE
    activity: Activity
    gss_block: GSS
    bf_block: Beamformer

    bf_drop_context: bool

    stft_size: int
    stft_shift: int
    stft_fading: bool

    # context_samples: int
    # equal_start_context: bool

    context_samples: int  # e.g. 240000
    multiarray: bool

    @property
    def db(self):
        return self.activity.db

    def stft(self, x):
        from nara_wpe.utils import stft
        return stft(
            x,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def istft(self, X):
        from nara_wpe.utils import istft
        return istft(
            X,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def get_iterator(self, session_id):
        return self.db.get_iterator_for_session(
            session_id,
            audio_read=False,
            adjust_times=False,  # not nessesary for chime6
            drop_unknown_target_speaker=True,
            context_samples=self.context_samples,
            equal_start_context=False,  # not nessesary for chime6
        )

    def enhance_session(
            self,
            session_ids,
            audio_dir,
            dataset_slice=False,
            audio_dir_exist_ok=False
    ):
        """

        Args:
            session_ids:
            audio_dir:
            dataset_slice:
            audio_dir_exist_ok:
                When True: It is ok, when the audio dir exists and the files
                insinde may be overwritten.

        Returns:


        >>> enhancer = get_enhancer(wpe=False, bss_iterations=2)
        >>> for x_hat in enhancer.enhance_session('S02'):
        ...     print(x_hat)
        """
        ensure_single_thread_numeric()

        audio_dir = Path(audio_dir)

        it = self.get_iterator(session_ids)

        if dlp_mpi.IS_MASTER:
            audio_dir.mkdir(exist_ok=audio_dir_exist_ok)

            for dataset in set(mapping.session_to_dataset.values()):
                (audio_dir / dataset).mkdir(exist_ok=audio_dir_exist_ok)

        dlp_mpi.barrier()

        if dataset_slice is not False:
            if dataset_slice is True:
                it = it[:2]
            elif isinstance(dataset_slice, int):
                it = it[:dataset_slice]
            elif isinstance(dataset_slice, slice):
                it = it[dataset_slice]
            else:
                raise ValueError(dataset_slice)

        for ex in dlp_mpi.split_managed(it, allow_single_worker=True):
            try:
                x_hat = self.enhance_example(ex)
                example_id = ex["example_id"]
                session_id = ex["session_id"]
                dataset = mapping.session_to_dataset[session_id]

                if x_hat.ndim == 1:
                    save_path = audio_dir / f'{dataset}' / f'{example_id}.wav'
                    dump_audio(
                        x_hat,
                        save_path,
                    )
                else:
                    raise NotImplementedError(x_hat.shape)
            except Exception:
                print('ERROR: Failed example:', ex['example_id'])
                raise

    def enhance_example(
            self,
            ex,
            debug=False,
    ):
        session_id = ex['session_id']
        speaker_id = ex['speaker_id']

        array_start = ex['start']
        array_end = ex['end']

        ex_array_activity = {
            # k: arr[array_start:min(array_end, len(arr))]
            k: arr[array_start:array_end]
            for k, arr in self.activity[session_id].items()
        }

        if self.multiarray is True:
            # ToDo: multiarray in ['except_wpe', 'only_wpe', ...]

            def concaternate_arrays(arrays):
                # The context does not consider the end of an utterance.
                # It can happen that some utterances are longer than others.
                # values = list(arrays.values())
                assert {v.ndim for v in arrays} == {2}, [v.shape for v in arrays]
                lengths = [v.shape[-1] for v in arrays]
                time_length = min(lengths)
                assert time_length > max(lengths) // 2, (time_length, lengths, ex['example_id'])
                values = [v[..., :time_length] for v in arrays]
                return np.array(values)

            obs = morph('ACN->A*CN', concaternate_arrays([
                load_audio(
                    ex['audio_path']['observation'][array],
                    start=ex['start'],
                    stop=ex['end'],
                )
                for array in sorted(ex['audio_path']['observation'].keys())
            ]))
        elif self.multiarray == 'outer_array_mics':
            def concaternate_arrays(arrays):
                # The context does not consider the end of an utterance.
                # It can happen that some utterances are longer than others.
                # values = list(arrays.values())
                assert {v.ndim for v in arrays} == {2}, [v.shape for v in arrays]
                time_length = min([v.shape[-1] for v in arrays])
                values = [v[(0, -1), :time_length] for v in arrays]
                return np.array(values)

            obs = morph('ACN->A*CN', concaternate_arrays([
                load_audio(
                    ex['audio_path']['observation'][array],
                    start=ex['start'],
                    stop=ex['end'],
                )
                for array in sorted(ex['audio_path']['observation'].keys())
            ]))
        elif self.multiarray == 'first_array_mics':
            def concaternate_arrays(arrays):
                # The context does not consider the end of an utterance.
                # It can happen that some utterances are longer than others.
                # values = list(arrays.values())
                assert {v.ndim for v in arrays} == {2}, [v.shape for v in arrays]
                time_length = min([v.shape[-1] for v in arrays])
                values = [v[(0,), :time_length] for v in arrays]
                return np.array(values)

            obs = morph('ACN->A*CN', concaternate_arrays([
                load_audio(
                    ex['audio_path']['observation'][array],
                    start=ex['start'],
                    stop=ex['end'],
                )
                for array in sorted(ex['audio_path']['observation'].keys())
            ]))
        elif self.multiarray is False:
            reference_array = ex['reference_array']
            obs = load_audio(
                ex['audio_path']['observation'][reference_array],
                start=ex['start'],
                stop=ex['end'],
            )
        else:
            raise ValueError(self.multiarray)

        x_hat = self.enhance_observation(
            obs,
            ex_array_activity=ex_array_activity,
            speaker_id=speaker_id,
            ex=ex,
            debug=debug,
        )

        if self.context_samples > 0:
            start_orig = ex['start_orig']
            start = ex['start']
            start_context = start_orig - start
            num_samples_orig = ex['num_samples_orig']
            x_hat = x_hat[..., start_context:start_context + num_samples_orig]
            # assert x_hat.shape[-1] == num_samples_orig, x_hat.shape
            # That assert does not work for P44_S18_U06_0265232-0265344.wav

        if debug:
            self.enhance_example_locals = locals()

        return x_hat

    def enhance_observation(
            self,
            obs,
            ex_array_activity,
            speaker_id,
            ex=None,
            debug=False,
    ):
        Obs = self.stft(obs)

        if self.wpe_block is not None:
            Obs = self.wpe_block(Obs, debug=debug)

        if self.gss_block.mm == 'NoisyGammacACGMM':
            assert list(ex_array_activity.keys())[-1] == 'Noise', (ex_array_activity.keys())

        acitivity_freq = activity_time_to_frequency(
            np.array(list(ex_array_activity.values())),
            stft_window_length=self.stft_size,
            stft_shift=self.stft_shift,
            stft_fading=self.stft_fading,
            stft_pad=True,
        )

        masks = self.gss_block(Obs, acitivity_freq, debug=debug)

        if self.bf_drop_context:
            start_context_frames, end_context_frames = start_end_context_frames(
                ex,
                stft_size=self.stft_size,
                stft_shift=self.stft_shift,
                stft_fading=self.stft_fading,
            )

            masks[:, :start_context_frames, :] = 0
            if end_context_frames > 0:
                masks[:, -end_context_frames:, :] = 0

        target_speaker_index = tuple(ex_array_activity.keys()).index(speaker_id)
        target_mask = masks[target_speaker_index]
        distortion_mask = np.sum(
            np.delete(masks, target_speaker_index, axis=0),
            axis=0,
        )




        X_hat = self.bf_block(
            Obs,
            target_mask=target_mask,
            distortion_mask=distortion_mask,
            debug=debug,
        )

        x_hat = self.istft(X_hat)

        if debug:
            self.enhance_observation_locals = locals()

        return x_hat


from pb_chime5.database.chime5.rttm import RTTMDatabase, get_chime6_files, groupby

def get_database():
    chime6_dir = Path('/net/fastdb/chime6/CHiME6')
    chime6_rttm = Path(
        '/net/vol/boeddeker/chime6/kaldi/egs/chime6/chime6_rttm')

    rttm = sorted(chime6_rttm.glob('*_rttm'))

    audio_paths = get_chime6_files(chime6_dir, worn=False, flat=True)
    audio_paths = {k: [p for p in paths if 'CH1' in p] for k, paths in
                        audio_paths.items()}
    alias = chime6_dir.glob('transcriptions/*/*.json')
    alias = groupby(sorted(alias), lambda p: p.parts[-2],
                         lambda p: p.with_suffix('').name)


def get_enhancer(
    multiarray=False,
    context_samples=240000,

    wpe=True,
    wpe_tabs=10,
    wpe_delay=2,
    wpe_iterations=3,
    wpe_psd_context=0,

    activity_type='annotation',  # ['annotation', 'path', 'non_sil_alignment']
    activity_path=None,
    activity_garbage_class=True,

    stft_size=1024,
    stft_shift=256,
    stft_fading=True,

    bss_iterations=20,
    bss_iterations_post=1,
    bss_mm='cACGMM',

    bf_drop_context=True,

    bf='mvdrSouden_ban',
    postfilter=None,

    database_path=str(JSON_PATH / 'chime6.json'),
):

    assert wpe is True or wpe is False, wpe

    assert activity_path is None or activity_type in ['path', 'rttm'], (activity_path, activity_type)

    return Enhancer(
        multiarray=multiarray,
        context_samples=context_samples,
        wpe_block=WPE(
            taps=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            psd_context=wpe_psd_context,
        ) if wpe else None,
        activity=Activity(
            type=activity_type,
            garbage_class=activity_garbage_class,
            path=activity_path,
            database_path=database_path,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
            mm=bss_mm,
            verbose=False,
        ),
        bf_drop_context=bf_drop_context,
        bf_block=Beamformer(
            type=bf,
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
    )
