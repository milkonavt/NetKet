# Import netket library
import netket as nk

print(nk.__version__)

# Import Json, this will be needed to load log files
# Import netket library
import netket as nk
import sys
# Helper libraries
import numpy as np


import netket.nn as nknn
import flax.linen as nn

import jax.numpy as jnp

import json

path_w='/n/home03/onikolaenko/NetKet/heisenberg/measurement'
# J2 = np.float32(sys.argv[1])
J2=0.5
# Couplings J1 and J2
J = [1, J2]
L = 14

# Define custom graph
edge_colors = []
for i in range(L):
    edge_colors.append([i, (i + 1) % L, 1])
    edge_colors.append([i, (i + 2) % L, 2])

# Define the netket graph object
g = nk.graph.Graph(edges=edge_colors)

# Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

bond_operator = [
    (J[0] * mszsz).tolist(),
    (J[1] * mszsz).tolist(),
    (-J[0] * exchange).tolist(),
    (J[1] * exchange).tolist(),
]

bond_color = [1, 2, 1, 2]


# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
op = nk.operator.GraphOperator(
    hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color
)


class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=2 * x.shape[-1],
            use_bias=True,
            param_dtype=np.complex128,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.normal(stddev=0.01),
        )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x


model = FFNN()

# We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=2)

# Construct the variational state
vs = nk.vqs.MCState(sa, model, n_samples=1008)

# We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
opt = nk.optimizer.Sgd(learning_rate=0.01)

# We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
# Note that we also specify the method to learn the parameters of the wave-function: here we choose the efficient
# Stochastic reconfiguration (Sr), here in an iterative setup
gs = nk.driver.VMC_SR(hamiltonian=op, optimizer=opt, variational_state=vs, diag_shift=0.01)

# We need to specify the local operators as a matrix acting on a local Hilbert space
sf = []
sites = []
structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
for i in range(0, L):
    for j in range(0, L):
        structure_factor += (
            (nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j))
            * ((-1) ** (i - j))
            / L
        )

# Run the optimization protocol
gs.run(out="test", n_iter=600, obs={"Structure Factor": structure_factor})

data = json.load(open("test.log"))

iters = data["Energy"]["iters"]
energy = data["Energy"]["Mean"]["real"]
sf = data["Structure Factor"]["Mean"]["real"]

np.savetxt('{path:s}/iters_J2={J2:.3f}.txt'.format(path=path_w,J2=J2),iters)
np.savetxt('{path:s}/energy_J2={J2:.3f}.txt'.format(path=path_w,J2=J2),energy)
np.savetxt('{path:s}/sf_J2={J2:.3f}.txt'.format(path=path_w,J2=J2),sf)