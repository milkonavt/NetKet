import os

import netket.operator

os.environ["NETKET_EXPERIMENTAL_SHARDING"]="1"


import netket as nk
import sys
import jax
import jax.numpy as jnp
import sys
print(jax.devices())
import numpy as np
import datetime
import time
import flax
from flax import linen as nn
from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd
from einops import rearrange

from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
path_w='/n/home03/onikolaenko/NetKet/heisenberg/measurement'
path_d='/n/home03/onikolaenko/NetKet/heisenberg/data'

from jax.nn.initializers import lecun_normal
from typing import Any

total_start_time = time.time()


print("he")
L = 2 # Side of the square
t=1;
U=0
n_fermions_per_spin = (1,1)

graph=nk.graph.Grid(extent=[L], pbc=True)
# graph.draw()
N = graph.n_nodes
Ne = sum(n_fermions_per_spin)
n_bands=2




hi = nk.hilbert.SpinOrbitalFermions(N, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin)


print("print all states")
print(hi.all_states())


seed = 0
key = jax.random.key(seed)
trial_state=hi.random_state(key,size=1)
print("trial state")
print(trial_state)


def hubbard_model(t, U, graph, hilbert):
    ham = 0.0
    for sz in (1,-1):
        for u, v in graph.edges():
            print("pair(u,v)=",u,v)
            ham += -t * cdag(hilbert, u, sz) * c(hilbert, v, sz) - t * cdag(hilbert, v, sz) * c(hilbert, u, sz)
    for u in graph.nodes():
        ham += U * nc(hilbert, u, 1) * nc(hilbert, u, -1)

    return ham.to_jax_operator()

ham_conn= hubbard_model(t, U, graph,        hi)
H = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(ham_conn)



# H1= H.to_dense()
# print("full hamiltonian")
# print(H1)
# a=cdag(hi,1, -1)*c(hi,1, -1)
# print("c,down",a.to_dense())
#

#Ground state energy

from scipy.sparse.linalg import eigsh
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=1, which="SA")

E_gs = eig_vals[0]

print("Exact ground state energy:", E_gs)


# print(np.kron(H1,np.eye(2)))

print("check manually constructing fermions")

print("*" * 80 )

# N_down: Total number of spin-up electrons
n_down_vec = np.array([0,  0, 1,  1])
# N_up: Total number of spin-down electrons
n_up_vec = np.array([0,  1,  0,  1])

# Number Operators
Nup = np.diag(n_up_vec)
Ndown = np.diag(n_down_vec)

# Jordan-Wigner Parity Operators: (-1)^N = I - 2N
JW_u = np.diag(1 - 2 * n_up_vec)
JW_d = np.diag(1 - 2 * n_down_vec)

JW = JW_u @ JW_d


tCu = np.zeros((4, 4))
tCu[0, 1] = tCu[2, 3] = 1
tCdu = np.transpose(tCu)

tCd = np.zeros((4, 4))
tCd[0, 2] = 1
tCd[1, 3] = -1
tCdd = np.transpose(tCd)

print("Cd= ",tCd)

print("Cu= ",tCu)

Cu = tCu
Cdu = tCdu
Cd = tCd
Cdd = tCdd

print("Cd= ",Cd)

print("Cu= ",Cu)


H = nk.operator.LocalOperator(hi)

print('L=',L)
for i in range(L):
    print("i=",i)
    H += nk.operator._local_operator.LocalOperator(hi, [U * Nup @ Ndown], [[0+i, L+i]])

for i in range(L-1):
    print("i=", i)
    H += -t * nk.operator._local_operator.LocalOperator(hi, [Cdu @ JW],[[0+i, L+i]]) * nk.operator._local_operator.LocalOperator(hi, [Cu],[[1+i,L+1+i]])
    H+=-t*nk.operator._local_operator.LocalOperator(hi,[Cdu],[[1+i,L+1+i]])*nk.operator._local_operator.LocalOperator(hi,[JW@Cu],[[0+i, L+i]])
    H+=-t*nk.operator._local_operator.LocalOperator(hi,[Cdd@JW],[[0+i, L+i]])*nk.operator._local_operator.LocalOperator(hi,[Cd],[[1+i,L+1+i]])
    H+=-t*nk.operator._local_operator.LocalOperator(hi,[Cdd],[[1+i,L+1+i]])*nk.operator._local_operator.LocalOperator(hi,[JW@Cd],[[0+i, L+i]])


print()
print("full hamiltonian")
print(H.to_dense())


sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

E_gs = eig_vals[0]

print("Exact ground state energy:", E_gs)
