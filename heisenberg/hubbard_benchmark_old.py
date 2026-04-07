import os

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

total_start_time = time.time()



L = 4 # Side of the square
graph = nk.graph.Square(L)
N = graph.n_nodes

t=1;
U=8


n_fermions_per_spin = (7,7)

Ne = sum(n_fermions_per_spin)

hi1= nk.hilbert.SpinOrbitalFermions(
    N, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
)

hi2 = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

hi=nk.hilbert.TensorHilbert(hi1,hi2)


seed = 0
key = jax.random.key(seed)

trial_state=hi.random_state(key,size=2)

# trial_state = jnp.array([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=jnp.int32)

now = datetime.datetime.now()





print("*" * 80 )
print("*" * 20 + " New calculation started. The local time is: " + "*" * 20)

print(f"System: {L}x{L} lattice with {N} sites")
print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
print(f"Parameters: t={t}, U={U}")
print()







# op = nc(hi, 1, 1)   # site 1, spin up
#
# for i, state in enumerate(trial_state):
#     xp, mels = op.get_conn(state)
#
#     print(f"\nState {i}:")
#     print("original state =", np.array(state))
#     print("result state   =", np.array(xp[0]))  # should be same state
#     print("eigenvalue     =", mels[0])

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
# Nf = hi.n_fermions
#
# # occupied orbital indices
# R = jnp.nonzero(trial_state, size=Nf)[0]
#
# # Slater submatrix (Nf x Nf) used by your class
# A = model.M[R, :]   # equivalent to self.M[R] inside the module
#
# print("State f =", np.array(trial_state, dtype=int))
# print("Occupied indices R =", np.array(R))
# print("Slater submatrix A, shape =", A.shape)
# print(np.array(A))
#
# # optional: check it matches the model output (log determinant)
# print("model(f) =", np.array(model(trial_state)))
# print("logdet(A) =", np.array(_logdet_cmplx(A)))
#
#


def extract_patches2d(x, patch_size):
    batch = x.shape[0]
    n_patches = int((x.shape[1] // patch_size**2) ** 0.5)
    x = x.reshape(batch, n_patches, patch_size, n_patches, patch_size)
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(batch, n_patches, n_patches, -1)
    x = x.reshape(batch, n_patches * n_patches, -1)
    return x


class Embed(nn.Module):
    d_model: int  # dimensionality of the embedding space
    patch_size: int  # linear patch size
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        x = extract_patches2d(x, self.patch_size)
        x = self.embed(x)

        return x

class Embed1(nn.Module):
    d_model: int        # embedding dimension
    n_sites: int        # L, number of sites
    n_bands: int = 2    # bits per site (up,down) => default 2
    param_dtype = jnp.float64

    def setup(self):
        # number of possible local states = 2**n_bands (e.g. 4 for up/down)
        self.embed = nn.Embed(
            num_embeddings=2 ** self.n_bands,
            features=self.d_model,
            embedding_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, n_flat):
        """
        n_flat: shape (2*n_sites,) or (batch, 2*n_sites), dtype 0/1 ints or floats.
        returns: embedded: (batch, n_sites, d_model) or (n_sites, d_model) if input was 1D.
        """
        n = jnp.asarray(n_flat)

        # reshape to (batch, n_sites, n_bands)

        down = n[:, :self.n_sites]  # first N entries
        up = n[:, self.n_sites:]  # last N entries

        # stack per site: (batch, N, 2)
        bits = jnp.stack([up, down], axis=-1)

        # build bit weights [1,2,...,2^(n_bands-1)]
        weights = (2 ** jnp.arange(self.n_bands)).astype(jnp.int32)  # shape (n_bands,)

        # compute integer token per site in [0, 2**n_bands - 1]
        tokens = jnp.sum(bits * weights, axis=-1).astype(jnp.int32)  # (batch, n_sites)

        # lookup embeddings: returns (batch, n_sites, d_model)
        embedded = self.embed(tokens)


        return embedded
# test embedding module implementation
d_model = 8  # embedding dimension
patch_size = 1  # linear patch size

# initialize a batch of spin configurations, considering a system on a 10x10 square lattice
M = 2


key, subkey = jax.random.split(key)
spin_configs = jax.random.randint(subkey, shape=(M, L * L), minval=0, maxval=1) * 2 - 1

spin_configs = jnp.array([
    [-1, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1,  1, 1, 1, 1, 1]
], dtype=jnp.int32)
print(f"{spin_configs.shape = }")
print(spin_configs)
# initialize the embedding module
embed_module = Embed(d_model, patch_size)

key, subkey = jax.random.split(key)
params_embed = embed_module.init(subkey, spin_configs)

# apply the embedding module to the spin configurations
embedded_configs = embed_module.apply(params_embed, spin_configs)

print(f"{embedded_configs.shape = }")


class Embed1(nn.Module):
    d_model: int  # embedding dimension
    n_sites: int  # L, number of sites
    n_bands: int = 2  # bits per site (up,down) => default 2
    param_dtype = jnp.float64

    def setup(self):
        # number of possible local states = 2**n_bands (e.g. 4 for up/down)
        self.embed = nn.Embed(
            num_embeddings=2 ** self.n_bands,
            features=self.d_model,
            embedding_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, n_flat):
        """
        n_flat: shape (2*n_sites,) or (batch, 2*n_sites), dtype 0/1 ints or floats.
        returns: embedded: (batch, n_sites, d_model) or (n_sites, d_model) if input was 1D.
        """
        n = jnp.asarray(n_flat)

        # reshape to (batch, n_sites, n_bands)

        down = n[:, :self.n_sites]  # first N entries
        up = n[:, self.n_sites:]  # last N entries

        # stack per site: (batch, N, 2)
        bits = jnp.stack([up, down], axis=-1)

        # build bit weights [1,2,...,2^(n_bands-1)]
        weights = (2 ** jnp.arange(self.n_bands)).astype(jnp.int32)  # shape (n_bands,)

        # compute integer token per site in [0, 2**n_bands - 1]
        tokens = jnp.sum(bits * weights, axis=-1).astype(jnp.int32)  # (batch, n_sites)

        # lookup embeddings: returns (batch, n_sites, d_model)
        embedded = self.embed(tokens)

        return embedded






embed_module = Embed1(d_model ,n_sites=L*L, n_bands=2)

key, subkey = jax.random.split(key)
params_embed = embed_module.init(subkey, trial_state)

# apply the embedding module to the spin configurations
embedded_configs = embed_module.apply(params_embed, trial_state)

print(f"{embedded_configs.shape = }")



class FactoredAttention(nn.Module):
    n_patches: int  # lenght of the input sequence
    d_model: int  # dimensionality of the embedding space (d in the equations)

    def setup(self):
        self.alpha = self.param(
            "alpha", nn.initializers.xavier_uniform(), (self.n_patches, self.n_patches)
        )
        self.V = self.param(
            "V", nn.initializers.xavier_uniform(), (self.d_model, self.d_model)
        )

    def __call__(self, x):
        y = jnp.einsum("i j, a b, M j b-> M i a", self.alpha, self.V, x)
        return y

from functools import partial


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        if self.transl_invariant:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches),
                self.param_dtype,
            )
            sq_n_patches = int(self.n_patches**0.5)
            assert sq_n_patches * sq_n_patches == self.n_patches
            self.alpha = roll2d(
                self.alpha, jnp.arange(sq_n_patches), jnp.arange(sq_n_patches)
            )
            self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
        else:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

    def __call__(self, x):
        # apply the value matrix in paralell for each head
        v = self.v(x)

        # split the representations of the different heads
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )

        # factored attention mechanism
        v = rearrange(
            v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
        )
        x = jnp.matmul(self.alpha, v)
        x = rearrange(
            x, "batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff"
        )

        # concatenate the different heads
        x = rearrange(
            x, "batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)"
        )

        # the representations of the different heads are combined together
        x = self.W(x)

        return x
# test Factored MultiHead Attention module
n_heads = 8  # number of heads
n_patches = embedded_configs.shape[1]   # lenght of the input sequence

# initialize the Factored Multi-Head Attention module
fmha_module = FMHA(d_model, n_heads, n_patches)

key, subkey = jax.random.split(key)
params_fmha = fmha_module.init(subkey, embedded_configs)

# apply the Factored Multi-Head Attention module to the embedding vectors
attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

print(f"{attention_vectors.shape = }")

class EncoderBlock(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x
# test Transformer Encoder module
num_layers = 4  # number of layers

# initialize the Factored Multi-Head Attention module
encoder_module = Encoder(num_layers, d_model, n_heads, n_patches)

key, subkey = jax.random.split(key)
params_encoder = encoder_module.init(subkey, embedded_configs)

# apply the Factored Multi-Head Attention module to the embedding vectors
x = embedded_configs
y = encoder_module.apply(params_encoder, x)
print("Multihead atteniton shape")
print(f"{y.shape = }")


log_cosh = (
    nk.nn.activation.log_cosh
)  # Logarithm of the hyperbolic cosine, implemented in a more stable way


class OuputHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j * out_imag

        return jnp.sum(log_cosh(out), axis=-1)


def compute_orbitals_fn(y, weights):
    return jnp.einsum( 'bnd,nde->bne',y, weights)


class OuputHeadDet(nn.Module):
    d_model: int
    Ne: int
    dtype: Any
    Ns: int



    def _log_det(self,A):

        sign, logabsdet = jnp.linalg.slogdet(A)
        cdtype = jnp.promote_types(A.dtype, jnp.complex64)
        amp = logabsdet.astype(cdtype) + jnp.log(sign.astype(cdtype))
        return jnp.where(jnp.isnan(amp), -jnp.inf, amp)
 # * shape = [batch, Ne]

    def setup(self):
        self.M1_up_real = self.param('M1_up_real', nn.initializers.xavier_uniform(), (self.Ns, self.d_model, self.Ne),
                                     self.dtype)
        self.M1_down_real = self.param('M1_down_real', nn.initializers.xavier_uniform(),
                                       (self.Ns, self.d_model, self.Ne), self.dtype)



    @nn.remat
    def __call__(self, y, R):
        orbitals_up_real = compute_orbitals_fn(y, self.M1_up_real)
        orbitals_down_real = compute_orbitals_fn(y, self.M1_down_real)

        orbitals_up = orbitals_up_real
        orbitals_down = orbitals_down_real

        orbitals = jnp.concatenate((orbitals_up, orbitals_down), axis=1)  # * shape = [batch, 2*Ns, Ne]



        # Find the positions of the occupied sites
        A = jnp.take_along_axis(orbitals, R[:, :, None], axis=1)  # * shape = [batch, Ne, Ne]

        log_det =self._log_det(A)
        return log_det


class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    Ne: int
    transl_invariant: bool = False


    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)

        Ns = x.shape[-1]//2  # number of sites
        n_patches = Ns // (self.patch_size**2)  # lenght of the input sequence
        f_non_zero = lambda x: x.nonzero(size=self.Ne)[0]
        R = jax.vmap(f_non_zero)(spins)  # * shape = [batch, Ne]

        # x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)
        x = Embed1(d_model=self.d_model, n_sites=L * L, n_bands=2)(x)
        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            transl_invariant=self.transl_invariant,
        )(x)

        log_psi = OuputHeadDet(d_model=self.d_model,Ne=self.Ne,Ns=Ns,dtype=jnp.float64)(y,R)

        return log_psi
# test ViT module

# initialize the ViT module

# vit_module = ViT(num_layers, d_model, n_heads, patch_size,Ne)
vit_module = ViT(
    num_layers=4, d_model=72, n_heads=12, patch_size=1, transl_invariant=True,Ne=Ne
)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, trial_state)

# apply the ViT module
log_psi = vit_module.apply(params, trial_state)

print(f"{log_psi.shape = }")



def hubbard_model(t, U, graph, hilbert):
    ham = 0.0
    for sz in (-1, 1):
        for u, v in graph.edges():
            ham += -t * cdag(hilbert, u, sz) * c(hilbert, v, sz) - t * cdag(hilbert, v, sz) * c(hilbert, u, sz)
    for u in graph.nodes():
        ham += U * nc(hilbert, u, 1) * nc(hilbert, u, -1)

    return ham.to_jax_operator()

ham_conn = hubbard_model(t, U, graph,        hi)



H = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(ham_conn)



# Define a Metropolis exchange sampler
N_samples = 4096
sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph,d_max=2,n_chains=N_samples)


optimizer = nk.optimizer.Sgd(learning_rate=0.01)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)


vmc = nk.driver.VMC_SR(
    hamiltonian=H,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="complex",
)

# Optimization
log = nk.logging.RuntimeLog()

N_opt = 2500
vmc.run(n_iter=N_opt, out=log)

full_energy = log.data["Energy"]["Mean"].real
print(f"Optimized energy : {full_energy}")
np.savetxt('{path:s}/energy_U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w,U=U,L=L,N_opt=N_opt,Ne=Ne),full_energy)

print('total time cost is', time.time() - total_start_time)