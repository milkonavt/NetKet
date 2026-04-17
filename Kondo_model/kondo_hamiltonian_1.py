import netket as nk
import numpy as np
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from netket.operator.spin import sigmaz, sigmap, sigmam
import netket.experimental as nkx


def make_graph_and_hilbert(Lx: int,Ly: int, n_fermions: int, pbc: bool = True):
    graph = nk.graph.Grid(extent=[Lx,Ly], pbc=pbc)
    N = graph.n_nodes
    hi_f = nk.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=n_fermions)
    hi_s = nk.hilbert.Spin(s=1/2, N=N)
    hi = nk.hilbert.TensorHilbert(hi_f, hi_s)
    return graph, N, hi_f, hi_s, hi



def build_fermion_hamiltonian(hi_f, hi, graph, t: float, U: float):
    terms = []
    weights = []

    # SpinOrbitalFermions uses orbital indices 0..hilbert.size-1.
    # For spinful fermions, NetKet's internal indexing is effectively
    # hi_f._get_index(site, sz) = spin_block * n_orbitals + site.
    def orb(site, sz):
        return hi_f._get_index(site, sz)

    for u, v in graph.edges():
        for sz in (+1, -1):
            iu = orb(u, sz)
            iv = orb(v, sz)

            # cdag(u,sz) @ c(v,sz)
            terms.append(((iu, 1), (iv, 0)))
            weights.append(-t)

            # cdag(v,sz) @ c(u,sz)
            terms.append(((iv, 1), (iu, 0)))
            weights.append(-t)

    for u in graph.nodes():
        iup = orb(u, +1)
        idn = orb(u, -1)

        # n_up n_dn = c†_up c_up c†_dn c_dn
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

    return nk.operator.EmbedOperator(hi, Hf, subspace=0)



def build_spin_hamiltonian(hi_s, hi, graph, J: float):
    Hs = 0.25 * nk.operator.Heisenberg(
        hilbert=hi_s,
        graph=graph,
        J=J,
        sign_rule=False,
    )
    return nk.operator.EmbedOperator(hi, Hs, subspace=1)





def build_kondo_hamiltonian_operator_based(hi_f, hi_s, hi, graph, Jk: float):
    Hk = nk.operator.LocalOperator(hi)

    for i in graph.nodes():
        # electron spin-density operators on fermion subspace
        s_e_z = 0.5 * (nc(hi_f, i, +1) - nc(hi_f, i, -1))
        s_e_p = cdag(hi_f, i, +1) @ c(hi_f, i, -1)
        s_e_m = cdag(hi_f, i, -1) @ c(hi_f, i, +1)

        # local-moment spin operators on spin subspace
        S_z = 0.5 * sigmaz(hi_s, i)   # physical S^z = sigma^z / 2
        S_p = sigmap(hi_s, i)         # physical S^+ = sigma^+
        S_m = sigmam(hi_s, i)         # physical S^- = sigma^-

        # embed each subsystem operator into the full TensorHilbert
        s_e_z_big = nk.operator.EmbedOperator(hi, s_e_z, subspace=0)
        s_e_p_big = nk.operator.EmbedOperator(hi, s_e_p, subspace=0)
        s_e_m_big = nk.operator.EmbedOperator(hi, s_e_m, subspace=0)

        S_z_big = nk.operator.EmbedOperator(hi, S_z, subspace=1)
        S_p_big = nk.operator.EmbedOperator(hi, S_p, subspace=1)
        S_m_big = nk.operator.EmbedOperator(hi, S_m, subspace=1)

        # Kondo term on site i
        Hk += Jk * (
            S_z_big @ s_e_z_big
            + 0.5 * S_p_big @ s_e_m_big
            + 0.5 * S_m_big @ s_e_p_big
        )

    return Hk



# def build_hamiltonian(hi_f, hi_s, hi, graph, t, U, J, Jk):
#     Hf_big = build_fermion_hamiltonian(hi_f, hi, graph, t=t, U=U)
#
#     # Hs_big = build_spin_hamiltonian(hi_s, hi, graph, J=J)
#     # SdotS = build_local_kondo_matrix()
#     # Hk = build_kondo_hamiltonian_operator_based(hi_f, hi_s, hi, graph, Jk=Jk)
#     #
#     # H = Hs_big + Hf_big + Hk
#
#     H = Hf_big
#
#     return H


#####New Method $


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

def build_spin_hamiltonian_subspace(hi_s, graph, J: float):
    return 0.25 * nk.operator.Heisenberg(
        hilbert=hi_s,
        graph=graph,
        J=J,
        sign_rule=False,
    )

def build_kondo_hamiltonian_full(hi_f, hi_s, hi, graph, Jk: float):
    Hk = nk.operator.LocalOperator(hi)

    for i in graph.nodes():
        s_e_z = 0.5 * (nc(hi_f, i, +1) - nc(hi_f, i, -1))
        s_e_p = cdag(hi_f, i, +1) @ c(hi_f, i, -1)
        s_e_m = cdag(hi_f, i, -1) @ c(hi_f, i, +1)

        S_z = 0.5 * sigmaz(hi_s, i)
        S_p = sigmap(hi_s, i)
        S_m = sigmam(hi_s, i)

        s_e_z_big = nk.operator.EmbedOperator(hi, s_e_z, subspace=0)
        s_e_p_big = nk.operator.EmbedOperator(hi, s_e_p, subspace=0)
        s_e_m_big = nk.operator.EmbedOperator(hi, s_e_m, subspace=0)

        S_z_big = nk.operator.EmbedOperator(hi, S_z, subspace=1)
        S_p_big = nk.operator.EmbedOperator(hi, S_p, subspace=1)
        S_m_big = nk.operator.EmbedOperator(hi, S_m, subspace=1)

        Hk += Jk * (
                S_z_big @ s_e_z_big
                + 0.5 * S_p_big @ s_e_m_big
                + 0.5 * S_m_big @ s_e_p_big
        )

    return Hk

from split_hamiltonian_1 import SplitKondoHamiltonian


def build_hamiltonian(hi_f, hi_s, hi, graph, t, U, J, Jk):
    Hf = build_fermion_hamiltonian_subspace(hi_f, graph, t=t, U=U)
    Hs = build_spin_hamiltonian_subspace(hi_s, graph, J=J)
    Hk = build_kondo_hamiltonian_full(hi_f, hi_s, hi, graph, Jk=Jk)

    return SplitKondoHamiltonian(
        hilbert=hi,
        hi_f=hi_f,
        hi_s=hi_s,
        Hf=Hf
    )

# def build_hamiltonian(hi_f, hi_s, hi, graph, t, U, J, Jk):
#     Hf_big = build_fermion_hamiltonian(hi_f, hi, graph, t=t, U=U)
#
#     # Hs_big = build_spin_hamiltonian(hi_s, hi, graph, J=J)
#     # SdotS = build_local_kondo_matrix()
#     # Hk = build_kondo_hamiltonian_operator_based(hi_f, hi_s, hi, graph, Jk=Jk)
#     #
#     # H = Hs_big + Hf_big + Hk
#
#     H = Hf_big
#
#     return H