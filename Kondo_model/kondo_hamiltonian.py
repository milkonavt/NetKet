import netket as nk
import numpy as np
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from netket.operator.spin import sigmaz, sigmap, sigmam


def make_graph_and_hilbert(Lx: int,Ly: int, n_fermions: int, pbc: bool = True):
    graph = nk.graph.Grid(extent=[Lx,Ly], pbc=pbc)
    N = graph.n_nodes
    hi_f = nk.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=n_fermions)
    hi_s = nk.hilbert.Spin(s=1/2, N=N)
    hi = nk.hilbert.TensorHilbert(hi_f, hi_s)
    return graph, N, hi_f, hi_s, hi

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

def build_spin_hamiltonian(hi_s, hi, graph, J: float):
    Hs = 0.25 * nk.operator.Heisenberg(
        hilbert=hi_s,
        graph=graph,
        J=J,
        sign_rule=False,
    )
    return nk.operator.EmbedOperator(hi, Hs, subspace=1)


def build_local_kondo_matrix() -> np.ndarray:
    # N_down: Total number of spin-up electrons
    n_down_vec = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # N_up: Total number of spin-down electrons
    n_up_vec = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    # Number Operators
    Nup = np.diag(n_up_vec)
    Ndown = np.diag(n_down_vec)

    tCu = np.zeros((4, 4))
    tCu[0, 1] = tCu[2, 3] = 1
    tCdu = np.transpose(tCu)

    tCd = np.zeros((4, 4))
    tCd[0, 2] = 1
    tCd[1, 3] = -1
    tCdd = np.transpose(tCd)

    Cu = np.kron(tCu, np.eye(2))
    Cdu = np.kron(tCdu, np.eye(2))
    Cd = np.kron(tCd, np.eye(2))
    Cdd = np.kron(tCdd, np.eye(2))

    tSz = np.zeros((2, 2))  # spin of local moment
    tSz[0, 0] = 0.5
    tSz[1, 1] = -0.5

    tSp = np.zeros((2, 2))
    tSp[0, 1] = 1.0

    tSm = np.zeros((2, 2))
    tSm[1, 0] = 1.0

    Sz = np.kron(np.eye(4), tSz)
    Sp = np.kron(np.eye(4), tSp)
    Sm = np.kron(np.eye(4), tSm)

    Sze = np.diag(0.5 * (n_up_vec - n_down_vec))
    Spe = np.dot(Cdu, Cd)
    Sme = np.dot(Cdd, Cu)
    Sxe = 0.5 * (Sp + Sm)
    Sye = -0.5j * (Sp - Sm)

    SdotS = np.matmul(Sz, Sze) + 0.5 * np.matmul(Sp, Sme) + 0.5 * np.matmul(Sm, Spe)

    return SdotS



def build_kondo_hamiltonian(hi, graph, Jk: float, SdotS: np.ndarray):
    Hk = nk.operator.LocalOperator(hi)
    N = graph.n_nodes

    for i in graph.nodes():
        Hk += nk.operator.LocalOperator(hi, [Jk*SdotS], [[0+i, N+i,2*N+i]])

    return Hk


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



def build_hamiltonian(hi_f, hi_s, hi, graph, t, U, J, Jk):
    Hf_big = build_fermion_hamiltonian(hi_f, hi, graph, t=t, U=U)

    Hs_big = build_spin_hamiltonian(hi_s, hi, graph, J=J)
    SdotS = build_local_kondo_matrix()
    Hk = build_kondo_hamiltonian_operator_based(hi_f, hi_s, hi, graph, Jk=Jk)

    H = Hs_big + Hf_big + Hk

    return H