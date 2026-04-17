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
import jax

def HaldaneHamiltonian(hi_f,
    nn_edges,
    nnn_edges,
    t1: float,
    t2: float,
    phi: float,
    sublattice,
    M: float = 0.0,
):
    """
    Spinless Haldane model:
        H = -t1 sum_<ij> (c_i^dag c_j + h.c.)
            -t2 sum_<<ij>> (exp(i * nu_ij * phi) c_i^dag c_j + h.c.)
            +M sum_i xi_i n_i

    Parameters
    ----------
    hi_f :
        Spinless fermionic Hilbert space.
    nn_edges : iterable of (u, v)
        Nearest-neighbor pairs.
    nnn_edges : iterable of (u, v, nu_uv)
        Directed next-nearest-neighbor pairs with nu_uv = ±1.
    t1 : float
        Nearest-neighbor hopping.
    t2 : float
        Next-nearest-neighbor hopping.
    phi : float
        Haldane phase.
    sublattice : sequence of ±1
        +1 on A sites, -1 on B sites.
    M : float
        Sublattice staggered potential.

    Returns
    -------
    H :
        NetKet particle-number-conserving fermionic operator.
    """

    terms = []
    weights = []

    # Nearest-neighbor hopping
    for u, v in nn_edges:
        print(f"paor{u,v}")
        terms.append(((u, 1), (v, 0)))
        weights.append(-t1)

        terms.append(((v, 1), (u, 0)))
        weights.append(-t1)

    # Next-nearest-neighbor complex hopping
    for u, v, nu_uv in nnn_edges:
        phase = np.exp(1j * nu_uv * phi)

        terms.append(((u, 1), (v, 0)))
        weights.append(-t2 * phase)

        terms.append(((v, 1), (u, 0)))
        weights.append(-t2 * np.conj(phase))

    # Sublattice mass term
    for i, xi in enumerate(sublattice):
        terms.append(((i, 1), (i, 0)))
        weights.append(M * xi)

    H_generic = nk.operator.FermionOperator2ndJax(
        hi_f,
        terms=terms,
        weights=weights,
    )

    H = nkx.operator.ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
        H_generic
    )

    return H