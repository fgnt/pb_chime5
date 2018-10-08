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

from cached_property import cached_property

import numpy as np

import nara_wpe
from dc_integration.distribution import (
    ComplexAngularCentralGaussianMixtureModel,
)
from dc_integration.distribution.utils import (
    stack_parameters,
)

from nt.utils.numpy_utils import morph
from nt.io.data_dir import database_jsons

from chime5.scripts.bss_inear_finetune import activity_time_to_frequency
from pb_chime5.io import load_audio


@dataclass
class WPE:
    K: int
    delay: int
    iterations: int
    neighborhood: int

    def __call__(self, Obs, stack=None):
        
        if Obs.ndim == 3:
            assert stack is None, stack
            Obs = nara_wpe.wpe.wpe_v8(
                Obs.transpose(2, 0, 1),
                K=self.K,
                delay=self.delay,
                iterations=self.iterations,
                neighborhood=self.neighborhood,
            ).transpose(1, 2, 0)
        elif Obs.ndim == 4:
            if stack is True:
                _A = Obs.shape[0]
                Obs = morph('ACTF->A*CTF', Obs)
                Obs = nara_wpe.wpe.wpe_v8(
                    Obs.transpose(2, 0, 1),
                    K=self.K,
                    delay=self.delay,
                    iterations=self.iterations,
                    neighborhood=self.neighborhood,
                ).transpose(1, 2, 0)
                Obs = morph('A*CTF->ACTF', Obs, A=_A)
            elif stack is False:
                Obs = nara_wpe.wpe.wpe_v8(
                    Obs.transpose(0, 3, 1, 2),
                    K=self.K,
                    delay=self.delay,
                    iterations=self.iterations,
                    neighborhood=self.neighborhood,
                ).transpose(0, 2, 3, 1)
                
            else:
                raise NotImplementedError(stack)
        else:
            raise NotImplementedError(Obs.shape)
        return Obs


@dataclass(hash=True)
class Activity:
    type: str = 'annotation'   # ['annotation', 'non_sil_alignment']
    garbage_class: bool = False
    # ali_path='/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali',
    # activity_array_ali_path='~/net/storage/jheymann/__share/jensheit/chime5/kaldi/arrayBSS_v5/exp/tri3_u_bss_js_cleaned_dev_new_bss_beam_39_ali',
    database_path: str = str(database_jsons / 'chime5_orig.json')

    @cached_property
    def db(self):
        from nt.database.chime5 import Chime5
        return Chime5(self.database_path)

    @staticmethod
    @functools.lru_cache(1)
    def _getitem(
            session_id,
            type,
            db,
            garbage_class,
    ):
        from pb_chime5.activity import get_activity

        assert type in ['annotation'], type

        return get_activity(
            iterator=db.get_iterator_by_names(session_id),
            perspective='array',
            garbage_class=garbage_class,
            dtype=np.bool,
            non_sil_alignment_fn=None,
            debug=False,
            use_ArrayIntervall=True,
        )[session_id]

    def __getitem__(self, session_id):
        return self._getitem(
            session_id,
            type=self.type,
            db=self.db,
            garbage_class=self.garbage_class,
        )


@dataclass
class GSS:
    iterations: int
    iterations_post: int

    verbose: bool = True

    use_pinv: bool = False
    stable: bool = True

    def __call__(self, Obs, acitivity_freq):

        initialization = np.asarray(acitivity_freq, dtype=np.float64)
        initialization = np.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / np.sum(initialization, keepdims=True,
                                                axis=0)
        initialization = np.repeat(initialization[None, ...], 513, axis=0)

        source_active_mask = np.asarray(acitivity_freq, dtype=np.bool)
        source_active_mask = np.repeat(source_active_mask[None, ...], 513, axis=0)

        cacGMM = ComplexAngularCentralGaussianMixtureModel(
            use_pinv=self.use_pinv,
            stable=self.stable,
        )

        learned = []
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
                Y=Obs.T[f, ...],
                initialization=initialization[f, ..., :T],
                iterations=self.iterations,
                source_activity_mask=source_active_mask[f, ..., :T],
            )

            if self.iterations_post != 0:
                cur = cacGMM.fit(
                    Y=Obs.T[f, ...],
                    initialization=cur,
                    iterations=self.iterations_post,
                )

            learned.append(cur)

        learned = stack_parameters(learned)
        posterior = learned.affiliation.transpose(1, 2, 0)
        return posterior


@dataclass
class Beamformer:
    type: str
    postfilter: str

    def __call__(self, Obs, target_mask, distortion_mask):
        bf = self.type

        if bf == 'mvdrSouden_ban':
            from chime5.enhancement.bf_lorenz import (
                beamform_mvdr_souden_from_masks,
                beamform_lcmv_souden_from_masks,
                beamform_gev_from_masks,
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

        return X_hat


@dataclass
class Enhancer:
    wpe_block: WPE
    activity: Activity
    gss_block: GSS
    bf_block: Beamformer

    stft_size: int
    stft_shift: int
    stft_fading: bool

    # context_samples: int
    # equal_start_context: bool

    context_samples = 0  # e.g. 240000
    multiarray = False

    dry_run = False

    @property
    def db(self):
        return self.activity.db

    def stft(self, x):
        from nt.transform import stft
        return stft(
            x,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def istft(self, X):
        from nt.transform import istft
        return istft(
            X,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def enhance_session(
            self,
            session_id,
    ):
        """
        >>> enhancer = get_enhancer(wpe=False, bss_iterations=2)
        >>> for x_hat in enhancer.enhance_session('S02'):
        ...     print(x_hat)
        """

        it = self.db.get_iterator_for_session(
            session_id,
            audio_read=False,
            adjust_times=True,
            drop_unknown_target_speaker=True,
            context_samples=self.context_samples,
            equal_start_context=True,
        )

        for ex in it[:2]:
            yield self.enhance_example(ex)


    def enhance_example(
            self,
            ex,
    ):
        session_id = ex['session_id']
        reference_array = ex['reference_array']
        speaker_id = ex['speaker_id']

        array_start = ex['start']['observation'][reference_array]
        array_end = ex['end']['observation'][reference_array]

        ex_array_activity = {
            k: arr[array_start:min(array_end, len(arr))]
            for k, arr in self.activity[session_id][reference_array].items()
        }

        if self.multiarray is True:
            obs = ...
            raise NotImplementedError(self.multiarray)
        elif self.multiarray is False:
            obs = load_audio(
                ex['audio_path']['observation'][reference_array],
                start=ex['start']['observation'][reference_array],
                stop=ex['end']['observation'][reference_array],
            )
        else:
            raise ValueError(self.multiarray)

        x_hat = self.enhance_observation(
            obs,
            ex_array_activity=ex_array_activity,
            speaker_id=speaker_id,
        )

        if self.context_samples > 0:
            start_orig = ex['start_orig']['observation'][reference_array]
            start = ex['start']['observation'][reference_array]
            start_context = start_orig - start
            num_samples_orig = ex['num_samples_orig']['observation'][reference_array]
            x_hat = x_hat[..., start_context:start_context + num_samples_orig]
            # assert x_hat.shape[-1] == num_samples_orig, x_hat.shape
            # That assert does not work for P44_S18_U06_0265232-0265344.wav

        return x_hat

    def enhance_observation(
            self,
            obs,
            ex_array_activity,
            speaker_id,
    ):
        Obs = self.stft(obs)

        if self.wpe_block is not None:
            Obs = self.wpe_block(Obs)

        acitivity_freq = activity_time_to_frequency(
            np.array(list(ex_array_activity.values())),
            stft_window_length=self.stft_size,
            stft_shift=self.stft_shift,
            stft_fading=self.stft_fading,
            stft_pad=True,
        )

        masks = self.gss_block(Obs, acitivity_freq)

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
        )

        x_hat = self.istft(X_hat)

        return x_hat


def get_enhancer(
    wpe=True,
    wpe_tabs=10,
    wpe_delay=3,
    wpe_iterations=3,
    wpe_neighborhood=1,

    activity_type='annotation',  # ['annotation', 'non_sil_alignment']
    activity_garbage_class=False,

    stft_size=1024,
    stft_shift=256,
    stft_fading=True,

    bss_iterations=20,
    bss_iterations_post=1,

    bf='mvdrSouden_ban',
    postfilter=None,
):
    return Enhancer(
        wpe_block=WPE(
            K=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            neighborhood=wpe_neighborhood,
        ) if wpe else None,
        activity=Activity(
            type=activity_type,
            garbage_class=activity_garbage_class,
            database_path=str(database_jsons / 'chime5_orig.json'),
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
            verbose=False,
            use_pinv=False,
            stable=True,
        ),
        bf_block=Beamformer(
            type=bf,
            postfilter=postfilter,
        ),

        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
    )
