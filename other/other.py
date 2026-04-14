def build_fermion_hamiltonian(hi_f, hi, graph, t: float, U: float):
    Hf = 0.0

    for u, v in graph.edges():
        for sz in (+1, -1):
            Hf += -t * (
                cdag(hi_f, u, sz) @ c(hi_f, v, sz)
                + cdag(hi_f, v, sz) @ c(hi_f, u, sz)
            )

    for u in graph.nodes():
        Hf += U * nc(hi_f, u, +1) @ nc(hi_f, u, -1)


    return nk.operator.EmbedOperator(hi, Hf, subspace=0)



def build_kondo_hamiltonian(hi, graph, Jk: float, SdotS: np.ndarray):
    Hk = nk.operator.LocalOperator(hi)
    N = graph.n_nodes

    for i in graph.nodes():
        Hk += nk.operator.LocalOperator(hi, [Jk*SdotS], [[0+i, N+i,2*N+i]])

    return Hk




###old split

import jax
import jax.numpy as jnp
import netket as nk
from netket.operator import AbstractOperator


class SplitKondoHamiltonian(AbstractOperator):
    def __init__(self, hilbert, hi_f, hi_s, Hf, Hs, Hk):
        self._hilbert = hilbert
        self.hi_f = hi_f
        self.hi_s = hi_s
        self.Hf = Hf
        self.Hs = Hs
        self.Hk = Hk

    @property
    def hilbert(self):
        return self._hilbert

    @property
    def dtype(self):
        return jnp.result_type(self.Hf.dtype, self.Hs.dtype, self.Hk.dtype)

    @property
    def is_hermitian(self):
        return True


def _combine_connected(op, x_full):
    Df = op.hi_f.size
    Ds = op.hi_s.size

    x_f = x_full[:, :Df]
    x_s = x_full[:, Df:Df + Ds]

    xp_f, mel_f = op.Hf.get_conn_padded(x_f)
    xp_s, mel_s = op.Hs.get_conn_padded(x_s)
    xp_k, mel_k = op.Hk.get_conn_padded(x_full)

    B = x_full.shape[0]

    Kf = xp_f.shape[1]
    x_s_pad = jnp.broadcast_to(x_s[:, None, :], (B, Kf, Ds))
    xp_f_full = jnp.concatenate([xp_f, x_s_pad], axis=-1)

    Ks = xp_s.shape[1]
    x_f_pad = jnp.broadcast_to(x_f[:, None, :], (B, Ks, Df))
    xp_s_full = jnp.concatenate([x_f_pad, xp_s], axis=-1)

    xp_all = jnp.concatenate([xp_f_full, xp_s_full, xp_k], axis=1)
    mel_all = jnp.concatenate([mel_f, mel_s, mel_k], axis=1)

    return xp_all, mel_all


def _chunked_apply_logpsi(logpsi, pars, x, chunk_size):
    """
    x: shape (N, D)
    returns shape (N,)
    """
    if chunk_size is None:
        return logpsi(pars, x)

    n = x.shape[0]
    outs = []

    for i in range(0, n, chunk_size):
        outs.append(logpsi(pars, x[i:i + chunk_size]))

    return jnp.concatenate(outs, axis=0)


def _local_kernel(logpsi, pars, sigma, extra_args, *, chunk_size=None):
    xp, mels = extra_args

    sigma_shape = sigma.shape
    D = sigma_shape[-1]
    K = xp.shape[-2]

    sigma_2d = sigma.reshape(-1, D)        # (B, D)
    xp_2d = xp.reshape(-1, K, D)           # (B, K, D)
    mels_2d = mels.reshape(-1, K)          # (B, K)

    # logpsi on original samples: chunk over batch B
    logpsi_sigma = _chunked_apply_logpsi(logpsi, pars, sigma_2d, chunk_size)

    # flatten connected states to evaluate psi(x') in chunks too
    xp_flat = xp_2d.reshape(-1, D)         # (B*K, D)
    logpsi_xp_flat = _chunked_apply_logpsi(logpsi, pars, xp_flat, chunk_size)
    logpsi_xp = logpsi_xp_flat.reshape(-1, K)  # (B, K)

    eloc = jnp.sum(mels_2d * jnp.exp(logpsi_xp - logpsi_sigma[:, None]), axis=-1)
    return eloc.reshape(sigma_shape[:-1])


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(
    vstate: nk.vqs.MCState,
    op: SplitKondoHamiltonian,
    chunk_size: int,
):
    return _local_kernel


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(
    vstate: nk.vqs.MCState,
    op: SplitKondoHamiltonian,
    chunk_size: type(None),
):
    return _local_kernel


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(
    vstate: nk.vqs.MCState,
    op: SplitKondoHamiltonian,
):
    sigma = vstate.samples
    sigma_shape = sigma.shape
    D = sigma_shape[-1]

    sigma_2d = sigma.reshape(-1, D)
    xp, mels = _combine_connected(op, sigma_2d)

    K = xp.shape[1]
    xp = xp.reshape(*sigma_shape[:-1], K, D)
    mels = mels.reshape(*sigma_shape[:-1], K)

    return sigma, (xp, mels)


##other vesrion

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


def _fermion_connected_only(op, x_full):
    """
    x_full: shape (B, Df + Ds)

    returns
      xp_full: shape (B, Kf, Df + Ds)
      mel_f:   shape (B, Kf)
    """

    Df = op.hi_f.size
    Ds = op.hi_s.size


    x_f = x_full[:, :Df]
    x_s = x_full[:, Df:Df + Ds]

    # connected states only from the fermionic operator
    xp_f, mel_f = op.Hf.get_conn_padded(x_f)



    B = x_full.shape[0]
    Kf = xp_f.shape[1]


    # keep the spin part unchanged
    x_s_pad = jnp.broadcast_to(x_s[:, None, :], (B, Kf, Ds))
    xp_full = jnp.concatenate([xp_f, x_s_pad], axis=-1)



    return xp_full, mel_f


def _chunked_apply_logpsi(logpsi, pars, x, chunk_size):
    """
    x: shape (N, D)
    returns shape (N,)
    """
    if chunk_size is None:
        return logpsi(pars, x)

    n = x.shape[0]
    outs = []
    for i in range(0, n, chunk_size):
        outs.append(logpsi(pars, x[i:i + chunk_size]))
    return jnp.concatenate(outs, axis=0)


def _local_kernel(logpsi, pars, sigma, extra_args, *, chunk_size=None):
    xp, mels = extra_args

    sigma_shape = sigma.shape
    D = sigma_shape[-1]
    K = xp.shape[-2]

    sigma_2d = sigma.reshape(-1, D)   # (B, D)
    xp_2d = xp.reshape(-1, K, D)      # (B, K, D)
    mels_2d = mels.reshape(-1, K)     # (B, K)

    logpsi_sigma = _chunked_apply_logpsi(logpsi, pars, sigma_2d, chunk_size)

    xp_flat = xp_2d.reshape(-1, D)    # (B*K, D)
    logpsi_xp_flat = _chunked_apply_logpsi(logpsi, pars, xp_flat, chunk_size)
    logpsi_xp = logpsi_xp_flat.reshape(-1, K)

    eloc = jnp.sum(mels_2d * jnp.exp(logpsi_xp - logpsi_sigma[:, None]), axis=-1)
    return eloc.reshape(sigma_shape[:-1])


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(
    vstate: nk.vqs.MCState,
    op: SplitKondoHamiltonian,
    chunk_size: int,
):
    return _local_kernel



@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(
    vstate: nk.vqs.MCState,
    op: SplitKondoHamiltonian,
):
    sigma = vstate.samples
    sigma_shape = sigma.shape
    D = sigma_shape[-1]

    sigma_2d = sigma.reshape(-1, D)

    xp, mels = _fermion_connected_only(op, sigma_2d)

    K = xp.shape[1]
    xp = xp.reshape(*sigma_shape[:-1], K, D)
    mels = mels.reshape(*sigma_shape[:-1], K)

    return sigma, (xp, mels)