import netket as nk
import os
import sys
import jax
import jax.numpy as jnp
import sys
print(jax.devices())
import numpy as np
import netket.experimental as nkx
from netket.operator import fermion
import flax
from flax import linen as nn
from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd
from einops import rearrange
path_w='/n/home03/onikolaenko/NetKet/heisenberg/measurement'
path_d='/n/home03/onikolaenko/NetKet/heisenberg/data'
L = 4  # Side of the square
graph = nk.graph.Square(L)
N = graph.n_nodes

# System parameters
Lx = 4  # Lattice width
Ly = 4  # Lattice height
t = 1.0  # Hopping strength
U = 8.0  # On-site interaction strength

g = nk.graph.Grid(extent=[Lx, Ly], pbc=True)
n_sites = g.n_nodes

# Create a hilbert space with spin-1/2 fermions
# 1/8 doping: 56 total fermions (28 spin-up, 28 spin-down)
n_fermions_per_spin = (7,7)
hi = nk.hilbert.SpinOrbitalFermions(
    n_sites, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
)

print(f"System: {Lx}x{Ly} lattice with {n_sites} sites")
print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
print(f"Parameters: t={t}, U={U}")
print()

# Create the Fermi-Hubbard Hamiltonian
# H = -t sum_{<ij>,sigma} (c_i^dag c_j + h.c.) + U sum_i (n_i_up n_i_down)
spin_values = [-1, 1]

# Hopping term
hop = sum(
    fermion.create(hi, site=i, sz=sz) @ fermion.destroy(hi, site=j, sz=sz)
    + fermion.create(hi, site=j, sz=sz) @ fermion.destroy(hi, site=i, sz=sz)
    for i, j in g.edges()
    for sz in spin_values
)

# On-site interaction term
interaction = sum(
    fermion.number(hi, site=i, sz=spin_values[0])
    @ fermion.number(hi, site=i, sz=spin_values[1])
    for i in range(n_sites)
)

ham = -t * hop + U * interaction
ham = ham.reduce()

# Initialize the mean-field variational state (Generalized Hartree-Fock)
vstate_mf = nkx.vqs.DeterminantVariationalState(hi, generalized=True, seed=42)

e_init = vstate_mf.expect(ham).mean
print(f"Initial energy: {e_init:.6f}")

print("\nOptimizing with gradient descent...")
opt = nk.optimizer.Sgd(learning_rate=0.1)
vmc = nk.driver.VMC(ham, optimizer=opt, variational_state=vstate_mf)
log = nk.logging.RuntimeLog()

N_opt=300
vmc.run(N_opt, out=log)

# Could convert to MCState:
# sampler = nk.sampler.MetropolisFermionHop(
#     hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
# )
# vstate_mc = vstate_mf.to_mcstate(sampler, n_samples=512, n_discard_per_chain=10)

energies = log.data["Energy"]["Mean"]
print(f"Final mean-field energy: {energies[-1]:.6f}")
energy_per_site = log.data["Energy"]["Mean"].real
np.savetxt('{path:s}/energy1_U={U:.3f}_L={L:d}_iter={N_opt:d}.txt'.format(path=path_w,U=U,L=L,N_opt=N_opt),energy_per_site)


outfile_path = '{path:s}/energy1_U={U:.3f}_L={L:d}_iter={N_opt:d}.txt'.format(path=path_d,U=U,L=L,N_opt=N_opt)

# Redirect stdout to file
sys.stdout = open(outfile_path, "a")