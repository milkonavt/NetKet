import os

import netket.operator

os.environ["NETKET_EXPERIMENTAL_SHARDING"]="1"


import netket as nk
import sys
import jax
import jax.numpy as jnp
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
path_w='/n/home03/onikolaenko/NetKet/Kondo_model/measurement'
path_d='/n/home03/onikolaenko/NetKet/Kondo_model/data'

from jax.nn.initializers import lecun_normal
from typing import Any
from netket.hilbert.constraint import DiscreteHilbertConstraint
total_start_time = time.time()
DType = Any
default_kernel_init = lecun_normal()

from kondo_hamiltonian import (
    make_graph_and_hilbert,
    build_hamiltonian,
)
from Embedding import Embed, FMHA, Encoder, ViT
from flax import struct

from sampler_rules import make_sz0_initial_states, KondoFlipRule, IdentityRule

L = 4
n_bands=3
t=1;
U=0;
Jk=1.5;
J=0.5;
Ne = 4


def main():
    graph, N, hi_f, hi_s, hi = make_graph_and_hilbert(L=L, n_fermions=Ne,pbc=True)

    # print("all states=",hi.all_states())
    seed = 0
    key = jax.random.key(seed)
    trial_state = hi.random_state(key, size=4)




    H = build_hamiltonian(hi_f, hi_s, hi, graph, t=t,U= U, J=J,Jk= Jk)


    # #Ground state energy
    sp_h = H.to_sparse()
    from scipy.sparse.linalg import eigsh

    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

    E_gs = eig_vals[0]

    print("Exact ground state energy:", E_gs)

    print("*" * 80)
    print("*" * 20 + " New calculation started. The local time is: " + "*" * 20)

    print(f"System: {L}x{L} lattice with {N} sites")
    print(f"Fermions: {Ne}")
    print(f"Parameters: t={t}, U={U}")
    print()

    embed_module = Embed(d_model=8, n_sites=N, n_bands=n_bands)

    key, subkey = jax.random.split(key)
    params_embed = embed_module.init(subkey, trial_state)  # create parameters

    # apply the embedding module to the trial configurations
    embedded_configs = embed_module.apply(params_embed, trial_state)

    print(f"{embedded_configs.shape = }")

    # test Factored MultiHead Attention module
    n_heads = 8  # number of heads
    d_model = 8
    n_patches = embedded_configs.shape[1]  # lenght of the input sequence

    # initialize the Factored Multi-Head Attention module
    fmha_module = FMHA(d_model, n_heads, n_patches)

    key, subkey = jax.random.split(key)
    params_fmha = fmha_module.init(subkey, embedded_configs)

    # apply the Factored Multi-Head Attention module to the embedding vectors
    attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

    print(f"{attention_vectors.shape = }")

    # test Transformer Encoder module
    num_layers = 4  # number of layers

    # initialize the Factored Multi-Head Attention module
    encoder_module = Encoder(num_layers, d_model, n_heads, n_patches)

    key, subkey = jax.random.split(key)
    params_encoder = encoder_module.init(subkey, embedded_configs)

    # apply the Factored Multi-Head Attention module to the embedding vectors
    x = embedded_configs
    y = encoder_module.apply(params_encoder, x)
    print("Multihead atteniton shape = ", y.shape)

    # vit_module = ViT(num_layers, d_model, n_heads, patch_size,Ne)
    vit_module = ViT(
        num_layers=2, d_model=8, n_heads=2, patch_size=1, transl_invariant=True, Ne=Ne, Ns=N, n_bands=n_bands
    )

    key, subkey = jax.random.split(key)
    params = vit_module.init(subkey, trial_state)

    # apply the ViT module
    log_psi = vit_module.apply(params, trial_state)

    print(f"{log_psi.shape = }")

    N_samples = 8192




    rule_f_sub = nk.sampler.rules.FermionHopRule(hi_f, graph=graph, d_max=2)
    rule_id_s = IdentityRule()
    rule_f = nk.sampler.rules.TensorRule(hi, (rule_f_sub, rule_id_s))

    rule_id_f = IdentityRule()
    rule_s_sub = nk.sampler.rules.ExchangeRule(
        graph=graph,
        d_max=2,  # or 1 for nearest-neighbor exchange
    )
    rule_s = nk.sampler.rules.TensorRule(hi, (rule_id_f, rule_s_sub))

    rule_k=KondoFlipRule(n_sites=N)

    rule_all = nk.sampler.rules.MultipleRules(
        rules=(rule_f, rule_s, rule_k),
        probabilities=jnp.array([1/3, 1/3, 1/3]),
    )




    sampler = nk.sampler.MetropolisSampler(
        hi,
        rule=rule_all,
        n_chains=8192,
    )


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



    sigma0 = make_sz0_initial_states(
        N=N,
        n_fermions=Ne,
        n_chains=sampler.n_chains,
        seed=1234,
    )


    vstate.sampler_state = vstate.sampler_state.replace(σ=sigma0)

    samples = vstate.sample(chain_length=16)  # shape: (n_chains, chain_length, 3*N)

    print("samples=",samples[1])

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

    N_opt = 10
    vmc.run(n_iter=N_opt, out=log)

    full_energy = log.data["Energy"]["Mean"].real
    print(f"Optimized energy : {full_energy}")
    np.savetxt(
        '{path:s}/energy_J={J:.3f}_Jk={Jk:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w, J=J,Jk=Jk, L=L, N_opt=N_opt,
                                                                                Ne=Ne), full_energy)

    # Save the state
    with open(
            "{path:s}/state_after_VMC_J={J:.3f}_Jk={Jk:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt".format(path=path_w, J=J,Jk=Jk, L=L, N_opt=N_opt,
                                                                                Ne=Ne),
            "wb",
    ) as file:
        file.write(flax.serialization.to_bytes(vstate))

    print('total time cost is', time.time() - total_start_time)


if __name__ == "__main__":
    main()







