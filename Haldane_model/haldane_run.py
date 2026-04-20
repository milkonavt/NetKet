import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
import matplotlib
matplotlib.use("TkAgg")  # or "MacOSX" on macOS

import matplotlib.pyplot as plt
import time
import datetime
from functools import partial
from pathlib import Path
from typing import Any

import jax
# jax.distributed.initialize()

import wandb
import yaml
import jax.numpy as jnp
import numpy as np
import netket as nk
import optax

from Embedding import Embed, FMHA, Encoder, ViT

from flax.serialization import to_bytes
from build_hamiltonian import HaldaneHamiltonian


def is_main_process():
    return jax.process_index() == 0


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dtype(dtype_name: str) -> Any:
    mapping = {
        "float32": jnp.float32,
        "float64": jnp.float64,
        "complex64": jnp.complex64,
        "complex128": jnp.complex128,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[dtype_name]


def maybe_init_wandb(cfg: dict):
    use_wandb = cfg.get("wandb", {}).get("use", False)
    if use_wandb and is_main_process():
        system_cfg = cfg["system"]
        wandb_cfg = cfg["wandb"]

        wandb.init(
            project=wandb_cfg["project"],
            name=(
                f"L={system_cfg['L']}_"
                f"t1={system_cfg['t1']}_"
                f"t2={system_cfg['t2']}_"
                f"phi={system_cfg['phi']}_"
                f"m={system_cfg['m']}_"
                f"N={system_cfg['n_particles']}"
            ),
            config=cfg,
        )


def maybe_log_wandb(cfg: dict, payload: dict):
    use_wandb = cfg.get("wandb", {}).get("use", False)
    if use_wandb and is_main_process():
        wandb.log(payload)


def shifted_cosine_decay(init_value, decay_steps, min_value=None):
    if min_value is None:
        min_value = init_value / 10.0

    cos_dec = optax.cosine_decay_schedule(
        init_value=init_value - min_value,
        decay_steps=decay_steps,
    )
    return lambda t: cos_dec(t) + min_value


def make_optimizer(cfg: dict):
    opt_cfg = cfg["optimizer"]
    vmc_cfg = cfg["vmc"]

    learning_rate = opt_cfg["learning_rate"]
    schedule_name = opt_cfg.get("schedule", "constant")

    if schedule_name == "cosine_decay":
        min_lr = opt_cfg.get("min_learning_rate", learning_rate / 10.0)
        lr_schedule = shifted_cosine_decay(
            init_value=learning_rate,
            decay_steps=vmc_cfg["n_iter"],
            min_value=min_lr,
        )
        optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
    else:
        lr_schedule = lambda step: learning_rate
        optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    return optimizer, lr_schedule




def make_wb_callback(n_sites: int, lr_schedule, cfg: dict):
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

        if is_main_process():
            msg = (
                f"step={step} "
                f"dt={dt:.3f}s/it "
                f"E={e.mean.real:.6f}"
            )
            if hasattr(e, "variance"):
                msg += f" var={e.variance.real:.6f}"
            print(msg, flush=True)

        if cfg.get("wandb", {}).get("use", False) and is_main_process():
            payload = {
                "energy per site": float(e.mean.real / n_sites),
                "iter_time_sec": dt,
                "learning_rate": current_lr,
                "iter_per_sec": inst_speed,
                "avg_iter_per_sec": avg_speed,
                "logpsi_time_sec": logpsi_time,
            }

            if hasattr(e, "variance"):
                payload["energy variance per site"] = float(e.variance.real / n_sites)

            wandb.log(payload)

        return True

    return wb_callback


def make_graph_and_hilbert(L: int, n_particles: int):
    graph = nk.graph.Honeycomb(extent=[L, L], pbc=True)
    N = graph.n_nodes

    hi = nk.hilbert.SpinOrbitalFermions(
        N,
        s=None,
        n_fermions=n_particles,
    )

    return graph, N, hi



def build_model(cfg: dict, Ne: int, Ns: int) -> ViT:
    model_cfg = cfg["model"]
    dtype = get_dtype(model_cfg.get("param_dtype", "float64"))

    return ViT(
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["embed_dim"],
        n_heads=model_cfg["n_heads"],
        patch_size=model_cfg["patch_size"],
        transl_invariant=model_cfg["transl_invariant"],
        Ne=Ne,
        Ns=Ns,
        n_bands=cfg["system"]["n_bands"],
        param_dtype=dtype,
    )


def run():
    logpsi=jnp.zeros((8192,12))
    print("timea a a",logpsi.shape)

    total_start_time = time.time()
    now = datetime.datetime.now()

    cfg = load_config("config.yaml")
    maybe_init_wandb(cfg)

    paths_cfg = cfg["paths"]

    if paths_cfg["cluster"]:
        output_dir = Path(paths_cfg["measurement_cluster"])
        data_dir = Path(paths_cfg["data_cluster"])
    else:
        output_dir = Path(paths_cfg["measurement_local"])
        data_dir = Path(paths_cfg["data_local"])

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    system_cfg = cfg["system"]
    L = system_cfg["L"]
    t1 = system_cfg["t1"]
    t2 = system_cfg["t2"]
    phi = np.pi * system_cfg["phi"]
    mass = system_cfg["m"]
    n_particles = system_cfg["n_particles"]
    n_bands = system_cfg["n_bands"]
    seed = system_cfg["seed"]

    graph, N, hi = make_graph_and_hilbert(L, n_particles)
    # graph.draw()
    Ne = n_particles

    key = jax.random.key(seed)
    trial_state = hi.random_state(key, size=1)
    print(trial_state.shape)

    print(trial_state)

    if is_main_process():
        print("trial_state =", trial_state)
        print("*" * 80)
        print("*" * 20 + f" New calculation started. The local time is: {now} " + "*" * 20)
        print(f"Honeycomb lattice: L={L}, number of sites={N}")
        print(f"Particles: N={n_particles}")
        print(f"Parameters: t1={t1}, t2={t2}, phi={phi}, m={mass}")
        print(f"n_bands={n_bands}")
        print()
    model_cfg = cfg["model"]
    embed_module = Embed(
        d_model=model_cfg["embed_dim"],
        n_sites=N,
        n_bands=1,
    )

    key, subkey = jax.random.split(key)
    params_embed = embed_module.init(subkey, trial_state)
    embedded_configs = embed_module.apply(params_embed, trial_state)

    if is_main_process():
        print(f"{embedded_configs.shape = }")

    n_patches = embedded_configs.shape[1]
    fmha_module = FMHA(72, 12, n_patches)

    key, subkey = jax.random.split(key)
    params_fmha = fmha_module.init(subkey, embedded_configs)
    attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

    if is_main_process():
        print(f"{attention_vectors.shape = }")

    encoder_module = Encoder(
        4,
        72,
        12,
        n_patches,
    )

    key, subkey = jax.random.split(key)
    params_encoder = encoder_module.init(subkey, embedded_configs)

    x = embedded_configs
    y = encoder_module.apply(params_encoder, x)

    if is_main_process():
        print("Multihead attention shape =", y.shape)










    model = build_model(cfg, Ne=Ne, Ns=N)

    key, subkey = jax.random.split(key)
    params = model.init(subkey, trial_state)

    log_psi = model.apply(params, trial_state)
    if is_main_process():
        print(f"{log_psi.shape = }")

    H = HaldaneHamiltonian(
        t1=t1,
        t2=t2,
        phi=phi,
        M=mass,
        graph=graph,
        hilbert=hi,
    )

    sampler_cfg = cfg["sampling"]
    sampler = nk.sampler.MetropolisFermionHop(
        hi,
        graph=graph,
        d_max=sampler_cfg["d_max"],
        n_chains=sampler_cfg["n_chains"],
    )

    optimizer, lr_schedule = make_optimizer(cfg)

    key, subkey = jax.random.split(key, 2)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        sampler_seed=subkey,
        n_samples=sampler_cfg["n_samples"],
        n_discard_per_chain=sampler_cfg["n_discard_per_chain"],
        variables=params,
        chunk_size=sampler_cfg["chunk_size"],
    )

    n_params = nk.jax.tree_size(vstate.parameters)
    if is_main_process():
        print("Number of parameters =", n_params, flush=True)

    vmc_cfg = cfg["vmc"]
    vmc = nk.driver.VMC_SR(
        hamiltonian=H,
        optimizer=optimizer,
        diag_shift=vmc_cfg["diag_shift"],
        variational_state=vstate,
        mode=vmc_cfg["mode"],
        on_the_fly=vmc_cfg["on_the_fly"],
        use_ntk=vmc_cfg["use_ntk"],
    )

    log = nk.logging.RuntimeLog()
    n_iter = vmc_cfg["n_iter"]
    callback = make_wb_callback(N, lr_schedule, cfg, H)

    vmc.run(
        n_iter=n_iter,
        out=log,
        callback=callback,
    )

    energy_history = np.asarray(log.data["Energy"]["Mean"].real)
    full_energy = energy_history[-1]
    runtime = time.time() - total_start_time

    if is_main_process():
        config_filename = (
            f"{output_dir}/config_"
            f"t1={t1:.3f}_t2={t2:.3f}_phi={phi:.3f}_m={m:.3f}_"
            f"L={L:d}_N={Ne:d}_iter={n_iter:d}.yaml"
        )

        with open(config_filename, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        state_filename = (
            f"{output_dir}/state_after_VMC_"
            f"t1={t1:.3f}_t2={t2:.3f}_phi={phi:.3f}_m={m:.3f}_"
            f"L={L:d}_N={Ne:d}_iter={n_iter:d}.mpack"
        )

        with open(state_filename, "wb") as file:
            file.write(to_bytes(vstate))

        energy_filename = (
            f"{output_dir}/energy_"
            f"t1={t1:.3f}_t2={t2:.3f}_phi={phi:.3f}_m={m:.3f}_"
            f"L={L:d}_N={Ne:d}_iter={n_iter:d}.txt"
        )
        np.savetxt(energy_filename, energy_history)

        print(f"Optimized energy : {full_energy}")
        print(f"energy per site : {full_energy / N}")
        print(f"number of parameters : {int(n_params)}")
        print("total time cost is", runtime)

    maybe_log_wandb(
        cfg,
        {
            "final_energy": float(full_energy),
            "final_energy_per_site": float(full_energy / N),
            "n_parameters": int(n_params),
            "runtime_sec": float(runtime),
        },
    )

    if cfg.get("wandb", {}).get("use", False) and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    run()