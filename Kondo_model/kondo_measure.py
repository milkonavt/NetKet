import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import sys
import time
import jax
# jax.distributed.initialize()

import wandb
import jax.numpy as jnp
import netket as nk
import netket.operator
import numpy as np
import flax
import optax
from flax.serialization import from_bytes

from kondo_hamiltonian import make_graph_and_hilbert, build_hamiltonian
from Embedding import Embed, FMHA, Encoder, ViT
from sampler_rules import make_sz0_initial_states, KondoFlipRule, IdentityRule

import experiment_config as conf

from jax.nn.initializers import lecun_normal
from typing import Any

DType = Any
default_kernel_init = lecun_normal()

print(f"process_index={jax.process_index()} / process_count={jax.process_count()}")
print(jax.devices())
print(sys.version)


def is_main_process():
    return jax.process_index() == 0


def maybe_init_wandb():
    if conf.use_wandb and is_main_process():
        wandb.init(
            project=conf.wandb_project,
            name=f"Lx={conf.Lx}_Ly={conf.Ly}_J={conf.J}_Jk={conf.Jk}_Ne={conf.Ne}",
            config=conf.cfg,
        )


def maybe_log_wandb(payload):
    if conf.use_wandb and is_main_process():
        wandb.log(payload)


def make_wb_callback(n_sites, lr_schedule):
    last_time = time.perf_counter()
    start_time = last_time

    def wb_callback(step, log_data, driver):
        nonlocal last_time
        current_lr = float(lr_schedule(step))

        now = time.perf_counter()
        dt = now - last_time
        total_time = now - start_time
        last_time = now

        e = log_data["Energy"]

        inst_speed = 1.0 / dt if dt > 0 else 0.0
        avg_speed = (step + 1) / total_time if total_time > 0 else 0.0

        # Print only from main process to avoid duplicated logs
        if is_main_process():
            msg = (
                f"step={step} "
                f"dt={dt:.3f}s/it "
                f"E={e.mean.real:.6f}"
            )
            if hasattr(e, "variance"):
                msg += f" var={e.variance.real:.6f}"

            print(msg, flush=True)

        # W&B only on main process
        if conf.use_wandb and is_main_process():
            payload = {
                "energy per site": float(e.mean.real / n_sites),
                "iter_time_sec": dt,
                "learning_rate": current_lr,
            }

            if hasattr(e, "variance"):
                payload["energy variance per site"] = float(e.variance.real / n_sites)

            wandb.log(payload)

        return True

    return wb_callback


def main():
    total_start_time = time.time()

    maybe_init_wandb()

    graph, N, hi_f, hi_s, hi = make_graph_and_hilbert(
        Lx=conf.Lx,
        Ly=conf.Ly,
        n_fermions=conf.Ne,
        pbc=conf.pbc,
    )

    key = jax.random.key(conf.seed)
    trial_state = hi.random_state(key, size=4)

    H = build_hamiltonian(
        hi_f,
        hi_s,
        hi,
        graph,
        t=conf.t,
        U=conf.U,
        J=conf.J,
        Jk=conf.Jk,
    )

    if is_main_process():
        print("*" * 80)
        print("*" * 20 + " New calculation started. The local time is: " + "*" * 20)
        print(f"System: {conf.Lx}x{conf.Ly} lattice with {N} sites")
        print(f"Fermions: {conf.Ne}")
        print(f"Parameters: t={conf.t}, U={conf.U}")
        print()

    embed_module = Embed(
        d_model=conf.d_model,
        n_sites=N,
        n_bands=conf.n_bands,
    )

    key, subkey = jax.random.split(key)
    params_embed = embed_module.init(subkey, trial_state)
    embedded_configs = embed_module.apply(params_embed, trial_state)

    if is_main_process():
        print(f"{embedded_configs.shape = }")

    n_patches = embedded_configs.shape[1]
    fmha_module = FMHA(conf.d_model, conf.n_heads, n_patches)

    key, subkey = jax.random.split(key)
    params_fmha = fmha_module.init(subkey, embedded_configs)
    attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

    if is_main_process():
        print(f"{attention_vectors.shape = }")

    encoder_module = Encoder(
        conf.num_layers,
        conf.d_model,
        conf.n_heads,
        n_patches,
    )

    key, subkey = jax.random.split(key)
    params_encoder = encoder_module.init(subkey, embedded_configs)

    x = embedded_configs
    y = encoder_module.apply(params_encoder, x)

    if is_main_process():
        print("Multihead attention shape =", y.shape)

    vit_module = ViT(
        num_layers=conf.num_layers,
        d_model=conf.d_model,
        n_heads=conf.n_heads,
        patch_size=conf.patch_size,
        transl_invariant=conf.transl_invariant,
        Ne=conf.Ne,
        Ns=N,
        n_bands=conf.n_bands,
    )

    key, subkey = jax.random.split(key)
    params = vit_module.init(subkey, trial_state)

    log_psi = vit_module.apply(params, trial_state)
    if is_main_process():
        print(f"{log_psi.shape = }")

    rule_f_sub = nk.sampler.rules.FermionHopRule(
        hi_f,
        graph=graph,
        d_max=conf.d_max,
    )
    rule_id_s = IdentityRule()
    rule_f = nk.sampler.rules.TensorRule(hi, (rule_f_sub, rule_id_s))

    rule_id_f = IdentityRule()
    rule_s_sub = nk.sampler.rules.ExchangeRule(
        graph=graph,
        d_max=conf.d_max,
    )
    rule_s = nk.sampler.rules.TensorRule(hi, (rule_id_f, rule_s_sub))

    rule_k = KondoFlipRule(n_sites=N)

    rule_all = nk.sampler.rules.MultipleRules(
        rules=(rule_f, rule_s, rule_k),
        probabilities=conf.rule_probabilities,
    )

    sampler = nk.sampler.MetropolisSampler(
        hi,
        rule=rule_all,
        n_chains=conf.n_chains,
    )

    def shifted_cosine_decay(init_value, decay_steps, min_value=None):
        if min_value is None:
            min_value = init_value / 10  # * reduce the lr by a factor 10

        cos_dec = optax.cosine_decay_schedule(init_value=init_value - min_value, decay_steps=decay_steps)
        lr_func = lambda t: cos_dec(t) + min_value  # shift the minimum value
        return lr_func

    lr_schedule = shifted_cosine_decay(
        init_value=conf.learning_rate,
        decay_steps=conf.N_opt,  # total number of VMC iterations
    )

    optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=conf.N_samples,
        n_discard_per_chain=conf.n_discard_per_chain,
        variables=params,
        chunk_size=conf.chunk_size,
    )

    state_filename = (
        f"{conf.path_w}/state_after_VMC_"
        f"J={conf.J:.3f}_Jk={conf.Jk:.3f}_"
        f"Lx={conf.Lx:d}_Ly={conf.Ly:d}_Ne={conf.Ne:d}_iter={conf.N_opt:d}.txt"
    )


    with open(
            state_filename,
            "rb",
    ) as file:
        vstate = from_bytes(vstate, file.read())

    order = ['E', 'CC', 'NN']

    if 'NN' in order:

        N = np.zeros((graph.n_nodes,), dtype=float)

        for i in graph.nodes():
            N_op = nc(hi, i, 1) + nc(hi, i, -1)
            N_op = N_op.to_jax_operator()
            N_op = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(N_op)
            N[i] = vstate.expect(N_op).mean.real

        print(N)
        np.savetxt('{path:s}/N_U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w, U=U, L=L,
                                                                                      N_opt=N_opt, Ne=Ne), N)

        NN_full = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)
        NN_conn = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)

        for i in graph.nodes():
            for j in graph.nodes():
                NN_op = (nc(hi, i, 1) + nc(hi, i, -1)) @ (nc(hi, j, 1) + nc(hi, j, -1))
                NN_op = NN_op.to_jax_operator()
                NN_op = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(NN_op)
                NN_full[i, j] = vstate.expect(NN_op).mean.real
                NN_conn[i, j] = NN_full[i, j] - N[i] * N[j]

        np.savetxt('{path:s}/NN_full_U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w, U=U, L=L,
                                                                                            N_opt=N_opt, Ne=Ne),
                   NN_full)

        np.savetxt('{path:s}/NN_U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w, U=U, L=L,
                                                                                       N_opt=N_opt, Ne=Ne), NN_conn)

    if 'CC' in order:

        CC = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)

        for i in graph.nodes():
            for j in graph.nodes():
                CC_op = (cdag(hi, i, 1)) @ (c(hi, j, 1))
                CC_op = CC_op.to_jax_operator()
                CC_op = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(CC_op)
                CC[i, j] = vstate.expect(CC_op).mean.real
                print(CC[i, j])

        np.savetxt('{path:s}/CC_U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={N_opt:d}.txt'.format(path=path_w, U=U, L=L,
                                                                                       N_opt=N_opt, Ne=Ne), CC)

        end = time.time()
        print(f"Execution time: {end - start:.3f} seconds")


if __name__ == "__main__":
    main()