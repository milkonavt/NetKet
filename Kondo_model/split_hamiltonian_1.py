import functools
import time

import jax
import jax.numpy as jnp
import netket as nk
from netket.operator import AbstractOperator


class SplitKondoHamiltonian(AbstractOperator):
    def __init__(self, hilbert, hi_f, hi_s, Hf):
        self._hilbert = hilbert
        self.hi_f = hi_f
        self.hi_s = hi_s
        self.Hf = Hf

    @property
    def hilbert(self):
        return self._hilbert

    @property
    def dtype(self):
        return self.Hf.dtype

    @property
    def is_hermitian(self):
        return True

    def fermion_connected_only(self, x_full):
        """
        x_full: array of shape (B, Df + Ds)

        Returns
        -------
        xp_full : array of shape (B, Kf, Df + Ds)
            Connected configurations where only the fermionic part changes.
        mel_f : array of shape (B, Kf)
            Matrix elements for the fermionic contribution.
        """
        Df = self.hi_f.size
        Ds = self.hi_s.size

        x_f = x_full[:, :Df]
        x_s = x_full[:, Df:Df + Ds]

        xp_f, mel_f = self.Hf.get_conn_padded(x_f)

        # xp_f has shape (B, Kf, Df)
        B, Kf, _ = xp_f.shape

        x_s_pad = jnp.broadcast_to(x_s[:, None, :], (B, Kf, Ds))
        xp_full = jnp.concatenate([xp_f, x_s_pad], axis=-1)


        return xp_full, mel_f

#
# @nk.vqs.get_local_kernel_arguments.dispatch
# def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: SplitKondoHamiltonian):
#     sigma = vstate.samples.reshape(-1, vstate.hilbert.size)
#     xp_full, mel_f = op.fermion_connected_only(sigma)
#     return sigma, (xp_full, mel_f)
#
#
# @nk.vqs.get_local_kernel.dispatch
# def get_local_kernel(vstate: nk.vqs.MCState, op: SplitKondoHamiltonian, chunk_size):
#     def local_kernel(logpsi, pars, sigma, extra_args, chunk_size=None):
#         xp_full, mel_f = extra_args
#
#         sigma = sigma.reshape(-1, vstate.hilbert.size)
#         logpsi_sigma = logpsi(pars, sigma)  # (B,)
#
#         B, Kf, D = xp_full.shape
#         xp_flat = xp_full.reshape(B * Kf, D)
#
#         # NetKet chunking here
#         logpsi_chunked = nk.jax.apply_chunked(
#             lambda p, x: logpsi(p, x),
#             in_axes=(None, 0),
#             chunk_size=chunk_size,
#         )
#
#         logpsi_xp_flat = logpsi_chunked(pars, xp_flat)
#         logpsi_xp = logpsi_xp_flat.reshape(B, Kf)
#
#         return jnp.sum(
#             mel_f * jnp.exp(logpsi_xp - logpsi_sigma[:, None]),
#             axis=1,
#         )
#
#     return local_kernel

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: SplitKondoHamiltonian):
    t0 = time.time()
    samples = vstate.samples
    jax.block_until_ready(samples)
    t1 = time.time()

    sigma = samples.reshape(-1, vstate.hilbert.size)
    jax.block_until_ready(sigma)
    t2 = time.time()

    print("sampling:", t1 - t0)
    print("reshape:", t2 - t1)

    time1 = time.time()
    xp_full, mel_f = op.fermion_connected_only(sigma)
    jax.block_until_ready(xp_full)
    time2=time.time()
    print(f"computing time: {time2-time1}")
    print(f"xp_full: {xp_full.shape}")
    print(f"mel_f: {mel_f.shape}")

    return sigma, (xp_full, mel_f)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: SplitKondoHamiltonian, chunk_size):
    def local_kernel(logpsi, pars, sigma, extra_args, chunk_size=None):
        xp_full, mel_f = extra_args
        print(f"xp_full: {xp_full.shape}")
        B, Kf, D = xp_full.shape

        logpsi_sigma = logpsi(pars, sigma)


        xp_flat = xp_full.reshape(B * Kf, D)

        logpsi_xp_flat = nk.jax.apply_chunked(
            logpsi,
            in_axes=(None, 0),
            chunk_size=chunk_size,
        )(pars, xp_flat)

        logpsi_xp = logpsi_xp_flat.reshape(B, Kf)
        print(f"logpsi: {logpsi_xp.shape}")
        val = jnp.sum(
            mel_f * jnp.exp(logpsi_xp - logpsi_sigma[:, None]),
            axis=1,
        )

        return val

    return local_kernel