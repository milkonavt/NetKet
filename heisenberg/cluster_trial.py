import netket as nk
import os
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
path_w='/n/home03/onikolaenko/NetKet/heisenberg/measurement'
path_d='/n/home03/onikolaenko/NetKet/heisenberg/data'

total_start_time = time.time()



L = 4  # Side of the square
graph = nk.graph.Square(L)
N = graph.n_nodes

t=1;
U=8


n_fermions_per_spin = (4, 4)
hi = nk.hilbert.SpinOrbitalFermions(
    N, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
)





now = datetime.datetime.now()




# outfile_path = '{path:s}/info_U={U:.3f}_L={L:d}.txt'.format(path=path_d,U=U,L=L)
# # Redirect stdout to file
# sys.stdout = open(outfile_path, "a")


print("*" * 80 )
print("*" * 20 + " New calculation started. The local time is: " + "*" * 20)

print(f"System: {L}x{L} lattice with {N} sites")
print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
print(f"Parameters: t={t}, U={U}")
print()



from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc

import netket as nk
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc


def hubbard_model(t, U, graph, hilbert):
    ham = 0.0
    for sz in (-1, 1):
        for u, v in graph.edges():
            ham += -t * cdag(hilbert, u, sz) * c(hilbert, v, sz) - t * cdag(hilbert, v, sz) * c(hilbert, u, sz)
    for u in graph.nodes():
        ham += U * nc(hilbert, u, 1) * nc(hilbert, u, -1)

    return ham.to_jax_operator()

ham_conn = hubbard_model(t, U, graph,        hi)

from netket.experimental.operator import ParticleNumberConservingFermioperator2nd


H = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(ham_conn)



# Convert the Hamiltonian to a sparse matrix
sp_h = H.to_sparse()

from scipy.sparse.linalg import eigsh

eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

E_gs = eig_vals[0]

print("Exact ground state energy:", E_gs)


# Convert the Hamiltonian to a sparse matrix
sp_h = H.to_sparse()

import jax
import jax.numpy as jnp


# Note: This function can also be found inside of netket, in `nk.jax.logdet_cmplx`, but we implement it here
# for pedagogical purposes.
def _logdet_cmplx(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


from flax import nnx
from jax.nn.initializers import lecun_normal
from typing import Any
from functools import partial

DType = Any
default_kernel_init = lecun_normal()

#
# class LogSlaterDeterminant(nnx.Module):
#     hilbert: nk.hilbert.SpinOrbitalFermions
#
#     def __init__(
#         self,
#         hilbert,
#         kernel_init=default_kernel_init,
#         param_dtype=float,
#         *,
#         rngs: nnx.Rngs,
#     ):
#         self.hilbert = hilbert
#
#         # To generate random numbers we need to extract the key from the `rngs` object.
#         key = rngs.params()
#
#         # the N x Nf matrix of the orbitals
#         self.M = nnx.Param(
#             kernel_init(
#                 key,
#                 (
#                     self.hilbert.local_size*self.hilbert.n_orbitals,
#                     self.hilbert.n_fermions,
#                 ),
#                 param_dtype,
#             )
#         )
#
#     def __call__(self, n: jax.Array) -> jax.Array:
#         # For simplicity, we write a function that operates on a single configuration of size (N,)
#         # and we vectorize it using `jnp.vectorize` with the signature='(n)->()' argument, which specifies
#         # that the function is defined to operate on arrays of shape (n,) and return scalars.
#         @partial(jnp.vectorize, signature="(n)->()")
#         def log_sd(n):
#             # Find the positions of the occupied orbitals
#             R = n.nonzero(size=self.hilbert.n_fermions)[0]
#
#             # Extract from the (N, Nf) matrix the (Nf, Nf) submatrix of M corresponding to the occupied orbitals.
#             A = self.M[R]
#
#             return _logdet_cmplx(A)
#
#         return log_sd(n)
#
#
# # Create the Slater determinant model, using the seed 0
# model = LogSlaterDeterminant(hi, rngs=nnx.Rngs(0))
#
# # Define the Metropolis-Hastings sampler
# sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)
#
#
# vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
#
# N_params = nk.jax.tree_size(vstate.parameters)
# print("Number of parameters = ", N_params, flush=True)
#
# # Define the optimizer
# op = nk.optimizer.Sgd(learning_rate=0.05)
#
# # Create the VMC (Variational Monte Carlo) driver with SR
# gs = nk.driver.VMC_SR(H, op, variational_state=vstate, diag_shift=0.05)
#
# # Construct the logger to visualize the data later on
# slater_log = nk.logging.RuntimeLog()
#
# # Run the optimization for 300 iterations
# N_opt=300
# gs.run(n_iter=300, out=slater_log)
#
# energy_per_site = slater_log.data["Energy"]["Mean"].real
# np.savetxt('{path:s}/energy_U={U:.3f}_L={L:d}_iter={N_opt:d}.txt'.format(path=path_w,U=U,L=L,N_opt=N_opt),energy_per_site)
# sd_energy = vstate.expect(H)
# # error = abs((sd_energy.mean - E_gs) / E_gs)
#
# print(f"Optimized energy : {sd_energy}")
# # print(f"Relative error   : {error}")


class LogNeuralBackflow(nnx.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions

    def __init__(
        self,
        hilbert,
        hidden_units: int,
        kernel_init=default_kernel_init,
        param_dtype=float,
        *,
        rngs: nnx.Rngs,
    ):
        self.hilbert = hilbert

        # To generate random numbers we need to extract the key from the `rngs` object.
        key = rngs.params()

        # the N x Nf matrix of the orbitals
        self.M = nnx.Param(
            kernel_init(
                key,
                (
                    self.hilbert.local_size*self.hilbert.n_orbitals,
                    self.hilbert.n_fermions,
                ),
                param_dtype,
            )
        )

        # Construct the Backflow. Takes as input strings of $N$ occupation numbers, outputs an $N x Nf$ matrix
        # that modifies the bare orbitals.
        self.backflow = nnx.Sequential(
            # First layer, input (..., N,) output (..., hidden_units)
            nnx.Linear(
                in_features=hilbert.size,
                out_features=hidden_units,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            nnx.tanh,
            nnx.Linear(
                in_features=hidden_units,
                out_features=hidden_units,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            nnx.tanh,
            # Last layer, input (..., hidden_units,) output (..., N x Nf)
            nnx.Linear(
                in_features=hidden_units,
                out_features=hidden_units,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            nnx.tanh,
            # Last layer, input (..., hidden_units,) output (..., N x Nf)
            nnx.Linear(
                in_features=hidden_units,
                out_features=hilbert.local_size*hilbert.n_orbitals * hilbert.n_fermions,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            # Reshape into the orbital shape, (..., N, Nf)
            lambda x: x.reshape(
                x.shape[:-1] + (hilbert.local_size*hilbert.n_orbitals, hilbert.n_fermions)
            ),
        )

    def __call__(self, n: jax.Array) -> jax.Array:
        # For simplicity, we write a function that operates on a single configuration of size (N,)
        # and we vectorize it using `jnp.vectorize` with the signature='(n)->()' argument, which specifies
        # that the function is defined to operate on arrays of shape (n,) and return scalars.
        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            # Construct the Backflow. Takes as input strings of $N$ occupation numbers, outputs an $N x Nf$ matrix
            # that modifies the bare orbitals.
            F = self.backflow(n)
            # Add the backflow correction to the bare orbitals
            M = self.M + F

            # Find the positions of the occupied, backflow-modified orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            A = M[R]
            return _logdet_cmplx(A)

        return log_sd(n)

# Create a neural backflow wave function
model = LogNeuralBackflow(hi, hidden_units=N, rngs=nnx.Rngs(3))

# Define a Metropolis exchange sampler
sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)

# Define an optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

# Create a variational state
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)


# Create a Variational Monte Carlo driver
gs = nk.driver.VMC_SR(H, op, variational_state=vstate, diag_shift=0.05)

# Construct the logger to visualize the data later on
bf_log = nk.logging.RuntimeLog()

# Run the optimization for 300 iterations

N_opt=800
gs.run(N_opt, out=bf_log)


energy_per_site = bf_log.data["Energy"]["Mean"].real
np.savetxt('{path:s}/energy1_U={U:.3f}_L={L:d}_iter={N_opt:d}.txt'.format(path=path_w,U=U,L=L,N_opt=N_opt),energy_per_site)
sd_energy = vstate.expect(H)
# error = abs((sd_energy.mean - E_gs) / E_gs)

print(f"Optimized energy : {sd_energy}")
# print(f"Relative error   : {error}")




print('total time cost is', time.time() - total_start_time)