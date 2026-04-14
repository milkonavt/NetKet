from datetime import time

import netket as nk
import numpy as np
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from netket.operator.spin import sigmaz, sigmap, sigmam
import netket.experimental as nkx
from netket.operator import AbstractOperator
import jax.numpy as jnp
import time

def build_fermion_hamiltonian_subspace(hi_f, graph, t: float, U: float):
    terms = []
    weights = []

    def orb(site, sz):
        return hi_f._get_index(site, sz)

    for u, v in graph.edges():
        for sz in (+1, -1):
            iu = orb(u, sz)
            iv = orb(v, sz)

            terms.append(((iu, 1), (iv, 0)))
            weights.append(-t)

            terms.append(((iv, 1), (iu, 0)))
            weights.append(-t)

    for u in graph.nodes():
        iup = orb(u, +1)
        idn = orb(u, -1)

        terms.append(((iup, 1), (iup, 0), (idn, 1), (idn, 0)))
        weights.append(U)

    Hf_generic = nk.operator.FermionOperator2ndJax(
        hi_f,
        terms=terms,
        weights=weights,
    )

    Hf = nkx.operator.ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
        Hf_generic
    )
    return Hf


class HubbardHamiltonian(AbstractOperator):
    def __init__(self, t,U, graph,hilbert):
        self.t = t
        self.U = U
        self.graph = graph
        self._hilbert = hilbert
        self.Hf = build_fermion_hamiltonian_subspace(
            self._hilbert, self.graph, self.t, self.U
        )


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

        xp_full, mel_f = self.Hf.get_conn_padded(x_full)


        return xp_full, mel_f


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: HubbardHamiltonian):

    sigma = vstate.samples.reshape(-1, vstate.hilbert.size)
    time1=time.time()
    xp_full, mel_f = op.fermion_connected_only(sigma)
    time2=time.time()
    print(f"computing time: {time2-time1}")
    return sigma, (xp_full, mel_f)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: HubbardHamiltonian, chunk_size):
    def local_kernel(logpsi, pars, sigma, extra_args, chunk_size=None):

        xp_full, mel_f = extra_args
        print(f"xp_full: {xp_full.shape}")

        logpsi_sigma = logpsi(pars, sigma)

        B, Kf, D = xp_full.shape
        xp_flat = xp_full.reshape(B * Kf, D)

        logpsi_xp_flat = nk.jax.apply_chunked(
            logpsi,
            in_axes=(None, 0),
            chunk_size=chunk_size,
        )(pars, xp_flat)

        # print(f"shape={logpsi_xp_flat.shape}")
        # print(f"chunksize={chunk_size}")
        # print(f"mel={mel_f.shape}")
        # exit()

        logpsi_xp = logpsi_xp_flat.reshape(B, Kf)

        print(f"logpsi: {logpsi_xp.shape}")

        loc_energy = jnp.sum(
            mel_f * jnp.exp(logpsi_xp - logpsi_sigma[:, None]),
            axis=1,
        )


        return loc_energy

    return local_kernel