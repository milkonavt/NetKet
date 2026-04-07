import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import sys
import time
import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import flax
import netket as nk
import netket.operator

from flax import linen as nn
from einops import rearrange
from jax.nn.initializers import lecun_normal
from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd
from netket.hilbert.constraint import DiscreteHilbertConstraint
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from scipy.sparse.linalg import eigsh

print(jax.devices())

path_w = "/n/home03/onikolaenko/NetKet/heisenberg/measurement"
path_d = "/n/home03/onikolaenko/NetKet/heisenberg/data"

total_start_time = time.time()


class TotalSzZeroConstraint(DiscreteHilbertConstraint):
    def __call__(self, x):
        # x shape: (..., total_number_of_dofs)

        # Adjust these slices to your actual TensorHilbert ordering
        # For SpinOrbitalFermions with spin-1/2, the fermionic part is occupations 0/1.
        # Suppose first 2*N entries are fermions, last N are local spins.
        down = x[:, :self.n_sites]  # first N entries
        print(down)

        up = x[:, self.n_sites : 2 * self.n_sites]  # next N entries
        print(up)

        spins = x[:, 2 * self.n_sites :]
        print(spins)

        # For nk.hilbert.Spin(s=1/2), local values are usually +/-1
        # so total Sz = 0.5 * sum(s)
        total_sz_twice = (
            jnp.sum(up, axis=-1) - jnp.sum(down, axis=-1) + jnp.sum(spins, axis=-1)
        )

        return total_sz_twice == 0


L = 2
n_bands = 3
t = 1
U = 0
Jk = 1.5
J = 0.0
n_fermions_per_spin = (1, 1)

graph = nk.graph.Grid(extent=[L], pbc=True)
N = graph.n_nodes
Ne = sum(n_fermions_per_spin)

hi_f = nk.hilbert.SpinOrbitalFermions(N, s=1 / 2, n_fermions=Ne)

hi_f= nk.hilbert.SpinOrbitalFermions(
    N, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
)
print("all_states=",hi_f.all_states())
Hf = 0.0
for u, v in graph.edges():
    for sz in (+1, -1):
        Hf += -t * (
            cdag(hi_f, u, sz) @ c(hi_f, v, sz)
            + cdag(hi_f, v, sz) @ c(hi_f, u, sz)
        )

for u in graph.nodes():
    Hf += U * nc(hi_f, u, +1) @ nc(hi_f, u, -1)

print("Hf=",Hf.to_dense())

#
# hi_s = nk.hilbert.Spin(s=1 / 2, N=N)
# hi = nk.hilbert.TensorHilbert(hi_f, hi_s)
#
# # print("all states=", hi.all_states())
#
# seed = 0
# key = jax.random.key(seed)
# trial_state = hi.random_state(key, size=4)
#
# Hf = 0.0
# for u, v in graph.edges():
#     for sz in (+1, -1):
#         Hf += -t * (
#             cdag(hi_f, u, sz) @ c(hi_f, v, sz)
#             + cdag(hi_f, v, sz) @ c(hi_f, u, sz)
#         )
#
# for u in graph.nodes():
#     Hf += U * nc(hi_f, u, +1) @ nc(hi_f, u, -1)
#
# Hf_big = nk.operator.EmbedOperator(hi, Hf, subspace=0)
#
# Hs = 1 / 4 * nk.operator.Heisenberg(
#     hilbert=hi_s,
#     graph=graph,
#     J=J,
#     sign_rule=[False, False],
# )
# Hs_big = nk.operator.EmbedOperator(hi, Hs, subspace=1)
#
# # N_down: Total number of spin-up electrons
# n_down_vec = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#
# # N_up: Total number of spin-down electrons
# n_up_vec = np.array([0, 0, 1, 1, 0, 0, 1, 1])
#
# # Number Operators
# Nup = np.diag(n_up_vec)
# Ndown = np.diag(n_down_vec)
#
# # Jordan-Wigner Parity Operators: (-1)^N = I - 2N
# JW_u = np.diag(1 - 2 * n_up_vec)
# JW_d = np.diag(1 - 2 * n_down_vec)
#
# tCu = np.zeros((4, 4))
# tCu[0, 1] = 1
# tCu[2, 3] = 1
# tCdu = np.transpose(tCu)
#
# tCd = np.zeros((4, 4))
# tCd[0, 2] = 1
# tCd[1, 3] = -1
# tCdd = np.transpose(tCd)
#
# Cu = np.kron(tCu, np.eye(2))
# Cdu = np.kron(tCdu, np.eye(2))
# Cd = np.kron(tCd, np.eye(2))
# Cdd = np.kron(tCdd, np.eye(2))
#
# tSz = np.zeros((2, 2))  # spin of local moment
# tSz[0, 0] = 0.5
# tSz[1, 1] = -0.5
#
# tSp = np.zeros((2, 2))
# tSp[0, 1] = 1.0
#
# tSm = np.zeros((2, 2))
# tSm[1, 0] = 1.0
#
# Sz = np.kron(np.eye(4), tSz)
# Sp = np.kron(np.eye(4), tSp)
# Sm = np.kron(np.eye(4), tSm)
#
# Sze = np.diag(0.5 * (n_up_vec - n_down_vec))
# Spe = np.dot(Cdu, Cd)
# Sme = np.dot(Cdd, Cu)
# Sxe = 0.5 * (Sp + Sm)
# Sye = -0.5j * (Sp - Sm)
#
# SdotS = np.matmul(Sz, Sze) + 0.5 * np.matmul(Sp, Sme) + 0.5 * np.matmul(Sm, Spe)
#
# Hk = nk.operator.LocalOperator(hi)
# for i in graph.nodes():
#     Hk += nk.operator._local_operator.LocalOperator(hi, [Jk * SdotS], [[0 + i, N + i, 2 * N + i]])
#
# H = Hs_big + Hf_big + Hk
#
# # Ground state energy
# sp_h = H.to_sparse()
# eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
# E_gs = eig_vals[0]
#
# print("Exact ground state energy:", E_gs)
#
# # rule_fermions = nk.sampler.rules.FermionHopRule(hi1, graph=graph, d=2)
# # rule_spins = nk.sampler.rules.LocalRule(hi2)
# # rules = [rule_fermions, rule_spins]
# # rule_fs = nk.sampler.rules.TensorRule(rules)
# # class (TwoLocalRule):
#
# print("*" * 80)
# print("*" * 20 + " New calculation started. The local time is: " + "*" * 20)
# print(f"System: {L}x{L} lattice with {N} sites")
# print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
# print(f"Parameters: t={t}, U={U}")
# print()
#
# DType = Any
# default_kernel_init = lecun_normal()
#
#
# class Embed(nn.Module):
#     d_model: int   # embedding dimension
#     n_sites: int   # number of sites
#     n_bands: int   # bits per site (up, down, spin) => here 3
#     param_dtype = jnp.float64
#
#     def setup(self):
#         # number of possible local states = 2**n_bands
#         self.embed = nn.Embed(
#             num_embeddings=2 ** self.n_bands,
#             features=self.d_model,
#             embedding_init=nn.initializers.xavier_uniform(),
#             param_dtype=self.param_dtype,
#         )
#
#     def __call__(self, n_flat):
#         """
#         n_flat: shape (2*n_sites,) or (batch, 2*n_sites), dtype 0/1 ints or floats.
#         returns: embedded: (batch, n_sites, d_model) or (n_sites, d_model) if input was 1D.
#         """
#         n = jnp.asarray(n_flat)
#
#         print("n= ", n)
#         print("hi")
#
#         # reshape to (batch, n_sites, n_bands)
#         down = n[:, :self.n_sites]  # first N entries
#         print(f"down={down}")
#
#         up = n[:, self.n_sites : 2 * self.n_sites]  # next N entries
#         print(f"up={up}")
#
#         spins = (n[:, 2 * self.n_sites :] + 1) // 2
#         print(f"spins={spins}")
#
#         # stack per site: (batch, N, 3)
#         bits = jnp.stack([up, down, spins], axis=-1)
#
#         # build bit weights [1,2,...,2^(n_bands-1)]
#         weights = (2 ** jnp.arange(self.n_bands)).astype(jnp.int32)
#
#         # compute integer token per site in [0, 2**n_bands - 1]
#         tokens = jnp.sum(bits * weights, axis=-1).astype(jnp.int32)
#         print(tokens)
#
#         # lookup embeddings: returns (batch, n_sites, d_model)
#         embedded = self.embed(tokens)
#
#         return embedded
#
#
# embed_module = Embed(d_model=8, n_sites=N, n_bands=n_bands)
#
# key, subkey = jax.random.split(key)
# params_embed = embed_module.init(subkey, trial_state)
#
# # apply the embedding module to the trial configurations
# embedded_configs = embed_module.apply(params_embed, trial_state)
#
# print(f"{embedded_configs.shape = }")