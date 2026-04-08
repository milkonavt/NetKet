import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange
from functools import partial
from typing import Any


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
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
            alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches),
                self.param_dtype,
            )
            sq_n_patches = int(self.n_patches ** 0.5)
            assert sq_n_patches * sq_n_patches == self.n_patches
            alpha = roll2d(
                alpha, jnp.arange(sq_n_patches), jnp.arange(sq_n_patches)
            )
            self.alpha = alpha.reshape(self.n_heads, -1, self.n_patches)
        else:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )
        v = rearrange(
            v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
        )
        x = jnp.matmul(self.alpha, v)
        x = rearrange(
            x, "batch n_heads n_patches d_eff -> batch n_patches n_heads d_eff"
        )
        x = rearrange(
            x, "batch n_patches n_heads d_eff -> batch n_patches (n_heads d_eff)"
        )
        x = self.W(x)
        return x


class Embed(nn.Module):
    d_model: int
    n_sites: int
    n_bands: int
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=2 ** self.n_bands,
            features=self.d_model,
            embedding_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, n_flat):
        n = jnp.asarray(n_flat)

        if n.ndim == 1:
            n = n[None, :]

        down = n[:, :self.n_sites]
        up = n[:, self.n_sites:2 * self.n_sites]
        spins = (n[:, 2 * self.n_sites:] + 1) // 2

        bits = jnp.stack([up, down, spins], axis=-1)
        weights = (2 ** jnp.arange(self.n_bands)).astype(jnp.int32)
        tokens = jnp.sum(bits * weights, axis=-1).astype(jnp.int32)

        embedded = self.embed(tokens)
        return embedded

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

def MSR_parity_fn_1layer(spins, Lx=None, Ly=None):
    """
    Compute MSR parity = N_down^A mod 2 for a single-layer 2D lattice.

    Parameters
    ----------
    spins : array-like, shape (batch, Ltot)
        Flattened configuration. This function assumes the physical spin sector
        is stored in the last third of the vector, i.e. spins[:, 2*L : 3*L].
    Lx, Ly : int, optional
        Lattice dimensions. If omitted, infer square lattice.

    Returns
    -------
    parity : array, shape (batch,)
        0 or 1
    """

    if spins.ndim != 2:
        raise ValueError(
            f"MSR_parity_fn_1layer expects spins with shape (batch, Ltot), got {spins.shape}"
        )

    B, Ltot = spins.shape

    if Ltot % 3 != 0:
        raise ValueError(
            f"Expected Ltot to be divisible by 3, got Ltot={Ltot} for shape {spins.shape}"
        )

    L = Ltot // 3

    # keep only the last third
    spins = spins[:, 2 * L :]   # shape (B, L)

    if spins.shape[1] != L:
        raise ValueError(
            f"After slicing expected shape (batch, {L}), got {spins.shape}"
        )

    if Lx is None or Ly is None:
        s = int(L ** 0.5)
        if s * s != L:
            raise ValueError(
                f"Cannot infer square lattice from L={L}. Please provide Lx and Ly."
            )
        Lx = Ly = s

    if Lx * Ly != L:
        raise ValueError(
            f"Lx*Ly must equal L. Got Lx={Lx}, Ly={Ly}, L={L}, shape={spins.shape}"
        )

    # reshape single-layer spins to (batch, x, y)
    spins_xy = rearrange(spins, "b (x y) -> b x y", x=Lx, y=Ly)

    xs = jnp.arange(Lx)[:, None]
    ys = jnp.arange(Ly)[None, :]
    mask_A = ((xs + ys) % 2) == 0   # A sublattice

    down_on_A = (spins_xy == -1) & mask_A
    N_down_A = down_on_A.sum(axis=(1, 2))

    return N_down_A % 2

class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    Ne: int #should I also introduce Ns here?
    Ns: int
    n_bands: int
    transl_invariant: bool = False


    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)


        Ns =self.Ns  # number of sites
        n_patches = Ns // (self.patch_size**2)  # lenght of the input sequence
        f_non_zero = lambda x: x.nonzero(size=self.Ne)[0]
        R = jax.vmap(f_non_zero)(spins)  # * shape = [batch, Ne]

        # x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)
        x = Embed(d_model=self.d_model, n_sites=Ns, n_bands=self.n_bands)(x)
        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            transl_invariant=self.transl_invariant,
        )(x)

        log_psi = OuputHeadDet(d_model=self.d_model,Ne=self.Ne,Ns=Ns,dtype=jnp.float64)(y,R)

        return log_psi+1j * jnp.pi * MSR_parity_fn_1layer(spins)