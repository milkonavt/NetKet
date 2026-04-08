import jax
jax.distributed.initialize()


import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"]="1"

import netket as nk

import sys

import jax.numpy as jnp
import sys

import numpy as np
import datetime
import time
from flax import linen as nn
from flax.serialization import to_bytes

from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd
from einops import rearrange
path_w='/n/home03/onikolaenko/NetKet/Hubbard/measurement'
path_d='/n/home03/onikolaenko/NetKet/Hubbard/data'


from jax.nn.initializers import lecun_normal
from typing import Any

from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc

total_start_time = time.time()
now = datetime.datetime.now()


L = 8  # Side of the square
t=1;
U=8
n_fermions_per_spin = (28,28)


graph = nk.graph.Square(L)
N = graph.n_nodes
Ne = sum(n_fermions_per_spin)
n_bands=2




hi = nk.hilbert.SpinOrbitalFermions(N, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin)



seed = 0
key = jax.random.key(seed)
trial_state=hi.random_state(key,size=1)
print("trial_state=", trial_state)



print("*" * 80 )
print("*" * 20 + " New calculation started. The local time is: " + "*" * 20)

print(f"System: {L}x{L} lattice with {N} sites")
print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
print(f"Parameters: t={t}, U={U}")
print()

DType = Any
default_kernel_init = lecun_normal()

class Embed(nn.Module):
    d_model: int  # embedding dimension
    n_sites: int  # L, number of sites
    n_bands: int   # bits per site (up,down) => default 2
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




embed_module = Embed(d_model=8 ,n_sites=L*L, n_bands=n_bands)


key, subkey = jax.random.split(key)
params_embed = embed_module.init(subkey, trial_state) # create parameters

# apply the embedding module to the trial configurations
embedded_configs = embed_module.apply(params_embed, trial_state)

print(f"{embedded_configs.shape = }")



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
d_model=8
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


def compute_orbitals_fn(y, weights):
    return jnp.einsum( 'bnd,nde->bne',y, weights)

def _log_det(A):

    sign, logabsdet = jnp.linalg.slogdet(A)
    cdtype = jnp.promote_types(A.dtype, jnp.complex64)
    amp = logabsdet.astype(cdtype) + jnp.log(sign.astype(cdtype))
    return jnp.where(jnp.isnan(amp), -jnp.inf, amp)

class OuputHeadDet(nn.Module):
    d_model: int
    Ne: int
    dtype: Any
    Ns: int




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

        log_det =_log_det(A)
        return log_det


class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    Ne: int #should I also introduce Ns here?
    Ns: int
    transl_invariant: bool = False



    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)


        Ns =self.Ns  # number of sites
        n_patches = Ns // (self.patch_size**2)  # lenght of the input sequence
        f_non_zero = lambda x: x.nonzero(size=self.Ne)[0]
        R = jax.vmap(f_non_zero)(spins)  # * shape = [batch, Ne]

        # x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)
        x = Embed(d_model=self.d_model, n_sites=Ns, n_bands=n_bands)(x)
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
    num_layers=4, d_model=72, n_heads=12, patch_size=1, transl_invariant=True,Ne=Ne,Ns=L*L
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
H=ham_conn


# Define a Metropolis exchange sampler
N_samples = 8192
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
    chunk_size=8192,
)




N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)



vmc = nk.driver.VMC_SR(
    hamiltonian=H,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="real",
    on_the_fly=True,
    use_ntk=True
)


# Optimization
log = nk.logging.RuntimeLog()

N_opt = 500
vmc.run(n_iter=N_opt, out=log)


full_energy = log.data["Energy"]["Mean"].real
print(f"Optimized energy : {full_energy}")
np.savetxt('{path:s}/energy_U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w,U=U,L=L,N_opt=N_opt,Ne=Ne),full_energy)

# Save the state
with open(
    "{path:s}/state_after_VMC.mpack".format(path=path_w,U=U,L=L,Ne=Ne),
    "wb",
) as file:
    file.write(to_bytes(vstate))



print('total time cost is', time.time() - total_start_time)