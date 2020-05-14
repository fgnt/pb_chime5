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

from pb_chime5.core_chime6 import WPE, GSS, start_end_context_frames, Beamformer

JSON_PATH = git_root / 'cache'


@dataclass  # (hash=True)
class Activity:
    garbage_class: bool = False
    rttm: str = None

    @cached_property
    def _data(self):
        from paderbox.array import intervall as array_intervall
        data = array_intervall.from_rttm(self.rttm)
        return data

    def __getitem__(self, session_id):
        from paderbox.array import intervall as array_intervall
        # todo: garbage class
        data = self._data

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
            data['Noise'] = array_intervall.ones()
        elif self.garbage_class is None:
            pass
        else:
            raise ValueError(self.garbage_class)

        return data


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

    db: 'RTTMDatabase'

    def stft(self, x):
        from paderbox.transform.module_stft import stft
        return stft(
            x,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def istft(self, X):
        from paderbox.transform.module_stft import istft
        return istft(
            X,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def get_dataset(self, session_id):
        return self.db.get_dataset_for_session(
            session_id,
            audio_read=True,
            adjust_times=False,  # not nessesary for chime6
            # drop_unknown_target_speaker=True,
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

        it = self.get_dataset(session_ids)

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

        obs = ex['audio_data']

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


def get_database(
    chime6_dir,
    rttm,
    multiarray,
):
    """
    >>> from paderbox.notebook import pprint
    >>> chime6_dir = Path('/net/fastdb/chime6/CHiME6')
    >>> rttm = Path('/net/vol/boeddeker/chime6/kaldi/egs/chime6/chime6_rttm')
    >>> rttm = sorted(rttm.glob('*_rttm'))
    >>> db = get_database(chime6_dir, rttm, 'first_array_mics')
    >>> pprint(db.get_dataset_for_session('S02', audio_read=True, context_samples=16000)[-1])
    {'example_id': 'S02_U06.-P05-142186400_142202560',
     'start': 142170400,
     'end': 142218560,
     'num_samples': 48160,
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
     'audio_data': array(shape=(6, 48160), dtype=float64)}
    """
    chime6_dir = Path(chime6_dir)

    if multiarray is True:
        audio_paths = get_chime6_files(chime6_dir, worn=False, flat=True)
    elif multiarray == 'outer_array_mics':
        audio_paths = get_chime6_files(chime6_dir, worn=False, flat=False)
        audio_paths = {
            session_id: [
                file
                for array_id, array_files in session_files.items()
                for file in [array_files[0], array_files[-1]]
            ]
            for session_id, session_files in audio_paths.items()
        }
    elif multiarray == 'first_array_mics':
        audio_paths = get_chime6_files(chime6_dir, worn=False, flat=False)
        audio_paths = {
            session_id: [
                array_files[0]
                for array_id, array_files in session_files.items()
            ]
            for session_id, session_files in audio_paths.items()
        }
    # elif multiarray.startswith('U'):
    #     # e.g. U01, U02, U03, U04, U05, U06
    #     audio_paths = get_chime6_files(chime6_dir, worn=False, flat=False)
    #     audio_paths = {
    #         session_id: session_files[multiarray]
    #         for session_id, session_files in audio_paths.items()
    #     }
    else:
        raise ValueError(multiarray)

    alias = chime6_dir.glob('transcriptions/*/*.json')
    alias = groupby(sorted(alias), lambda p: p.parts[-2],
                         lambda p: p.with_suffix('').name)

    return RTTMDatabase(
        rttm, audio_paths, alias=alias
    )


def get_enhancer(
    database_rttm,
    activity_rttm,
    chime6_dir='/net/fastdb/chime6/CHiME6',
    multiarray='outer_array_mics',
    context_samples=240000,

    wpe=True,
    wpe_tabs=10,
    wpe_delay=2,
    wpe_iterations=3,
    wpe_psd_context=0,

    activity_garbage_class=True,

    stft_size=1024,
    stft_shift=256,
    stft_fading=True,

    bss_iterations=20,
    bss_iterations_post=1,

    bf_drop_context=True,

    bf='mvdrSouden_ban',
    postfilter=None,
):

    assert wpe is True or wpe is False, wpe

    db = get_database(
        chime6_dir,
        database_rttm,
        multiarray,
    )

    return Enhancer(
        db=db,
        context_samples=context_samples,
        wpe_block=WPE(
            taps=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            psd_context=wpe_psd_context,
        ) if wpe else None,
        activity=Activity(
            garbage_class=activity_garbage_class,
            rttm=activity_rttm,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
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
