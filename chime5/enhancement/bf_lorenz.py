import numpy as np

from cached_property import cached_property

from nt.utils.numpy_utils import morph
from nt.speech_enhancement import beamformer
from nt.speech_enhancement.mask_module import lorenz_mask


class _Beamformer:
    def __init__(
            self,
            Y,
            X_mask,
            N_mask,
            debug=False,
    ):
        self.debug = debug

        if np.ndim(Y) == 4:
            self.Y = morph('1DTF->FDT', Y)
        else:
            self.Y = morph('DTF->FDT', Y)

        if np.ndim(X_mask) == 4:
            self.X_mask = morph('1DTF->FT', X_mask, reduce=np.median)
            self.N_mask = morph('1DTF->FT', N_mask, reduce=np.median)
        else:
            # elif np.ndim(X_mask) == 3:
            self.X_mask = morph('DTF->FT', X_mask, reduce=np.median)
            self.N_mask = morph('DTF->FT', N_mask, reduce=np.median)
        # elif np.ndim(X_mask) == 2:
        #     self.X_mask = morph('TF->FT', X_mask, reduce=np.median)
        #     self.N_mask = morph('TF->FT', N_mask, reduce=np.median)

        if self.debug:
            print('Y', repr(self.Y))
            print('X_mask', repr(self.X_mask), 'N_mask', repr(self.N_mask))

        assert self.Y.ndim == 3, self.Y.shape
        F, D, T = self.Y.shape
        assert D < 20, (D, self.Y.shape)
        assert self.X_mask.shape == (F, T), (self.X_mask.shape, F, T)
        assert self.N_mask.shape == (F, T), (self.N_mask.shape, F, T)


    @cached_property
    def _Cov_X(self):
        Cov_X = beamformer.get_power_spectral_density_matrix(self.Y, self.X_mask)
        if self.debug:
            print('Cov_X', repr(Cov_X))
        return Cov_X

    @cached_property
    def _Cov_N(self):
        Cov_N = beamformer.get_power_spectral_density_matrix(self.Y, self.N_mask)
        if self.debug:
            print('Cov_N', repr(Cov_N))
        return Cov_N

    @cached_property
    def _w_mvdr_souden(self):
        w_mvdr_souden = beamformer.get_mvdr_vector_souden(self._Cov_X, self._Cov_N, eps=1e-10)
        if self.debug:
            print('w_mvdr_souden', repr(w_mvdr_souden))
        return w_mvdr_souden

    @cached_property
    def X_hat_mvdr_souden(self):
        return beamformer.apply_beamforming_vector(self._w_mvdr_souden, self.Y).T


def beamform_mvdr_souden_from_masks(
        Y,
        X_mask,
        N_mask,
        debug=False,
):
    return _Beamformer(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
        debug=debug,
    ).X_hat_mvdr_souden


def beamform_mvdr_souden_with_lorenz_mask(
        Y,
        X_hat=None,
        debug=False,
):
    if X_hat is None:
        X_hat = Y

    X_mask = np.swapaxes(lorenz_mask(np.swapaxes(X_hat, -2, -1)), -2, -1)
    N_mask = 1 - X_mask

    return beamform_mvdr_souden_from_masks(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
        debug=debug,
    )
