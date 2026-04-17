import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
import yaml
import sys
import time
import jax
jax.distributed.initialize()

import wandb
import jax.numpy as jnp
import netket as nk
import netket.operator
import numpy as np
import flax
import optax

from kondo_hamiltonian_1 import make_graph_and_hilbert, build_hamiltonian
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

def benchmark_logpsi_current(vstate, H):
    sigma = vstate.samples.reshape(-1, vstate.hilbert.size)
    time1=time.time()
    xp_full, mel_f = H.fermion_connected_only(sigma)
    jax.block_until_ready(xp_full)
    time2 = time.time()
    print(f"computing time xpfull: {time2 - time1}")

    time1=time.time()
    B, Kf, D = xp_full.shape
    jax.block_until_ready(B)
    xp_flat = xp_full.reshape(B * Kf, D)

    f = nk.jax.apply_chunked(
        vstate._apply_fun,
        in_axes=(None, 0),
        chunk_size=vstate.chunk_size,
    )
    time2 = time.time()
    print(f"computing time reshaping: {time2 - time1}")



    t0 = time.perf_counter()
    y = f(vstate.variables, xp_flat)
    jax.block_until_ready(y)
    t1 = time.perf_counter()
    print(f"yshape={y.shape}")

    return t1 - t0
def make_wb_callback(n_sites,lr_schedule,hamiltonian):
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

        logpsi_time = 0

        inst_speed = 1.0 / dt if dt > 0 else 0.0
        avg_speed = (step + 1) / total_time if total_time > 0 else 0.0

        # Print only from main process to avoid duplicated logs
        if is_main_process():
            msg = (
                f"step={step} "
                f"dt={dt:.3f}s/it "
                f"logpsi={logpsi_time:.6f}s "
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
    trial_state = hi.random_state(key, size=5)

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

    key, subkey = jax.random.split(key, 2)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=conf.N_samples,
        n_discard_per_chain=conf.n_discard_per_chain,
        variables=params,
        chunk_size=conf.chunk_size,
    )

    sigma0 = make_sz0_initial_states(
        N=N,
        n_fermions=conf.Ne,
        n_chains=sampler.n_chains,
        seed=conf.init_seed,
    )
    vstate.sampler_state = vstate.sampler_state.replace(σ=sigma0)

    N_params = nk.jax.tree_size(vstate.parameters)
    if is_main_process():
        print("Number of parameters =", N_params, flush=True)

    vmc = nk.driver.VMC_SR(
        hamiltonian=H,
        optimizer=optimizer,
        diag_shift=conf.diag_shift,
        variational_state=vstate,
        mode=conf.mode,
        on_the_fly=conf.on_the_fly,
        use_ntk=conf.use_ntk,
    )

    log = nk.logging.RuntimeLog()
    callback = make_wb_callback(N,lr_schedule,H)
    vmc.run(
        n_iter=conf.N_opt,
        out=log,
        callback=callback,
    )

    energy_history = np.asarray(log.data["Energy"]["Mean"].real)
    full_energy = energy_history[-1]
    runtime = time.time() - total_start_time

    # Save only once
    if is_main_process():
        config_filename = (
            f"{conf.path_w}/config_"
            f"J={conf.J:.3f}_Jk={conf.Jk:.3f}_"
            f"Lx={conf.Lx:d}_Ly={conf.Ly:d}_Ne={conf.Ne:d}_iter={conf.N_opt:d}.yaml"
        )

        with open(config_filename, "w") as f:
            yaml.dump(conf.cfg, f)



        state_filename = (
            f"{conf.path_w}/state_after_VMC_"
            f"J={conf.J:.3f}_Jk={conf.Jk:.3f}_"
            f"Lx={conf.Lx:d}_Ly={conf.Ly:d}_Ne={conf.Ne:d}_iter={conf.N_opt:d}.txt"
        )

        with open(state_filename, "wb") as file:
            file.write(flax.serialization.to_bytes(vstate))

        energy_filename = (
            f"{conf.path_w}/energy_"
            f"J={conf.J:.3f}_Jk={conf.Jk:.3f}_"
            f"Lx={conf.Lx:d}_Ly={conf.Ly:d}_Ne={conf.Ne:d}_iter={conf.N_opt:d}.txt"
        )
        np.savetxt(energy_filename, energy_history)

        print(f"Optimized energy : {full_energy}")
        print(f"number of parameters : {int(N_params)}")
        print("total time cost is", runtime)





    # maybe_log_wandb(
    #     {
    #         "final_energy": float(full_energy),
    #         "n_parameters": int(N_params),
    #         "runtime": runtime,
    #     }
    # )


if __name__ == "__main__":
    main()