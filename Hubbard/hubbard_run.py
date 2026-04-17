
import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import jax
jax.distributed.initialize()

import time
import datetime
from pathlib import Path
from typing import Any

import wandb
import yaml
import jax.numpy as jnp
import numpy as np
import netket as nk
import optax

from Embedding import  ViT

from flax.serialization import to_bytes
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd
from build_hamiltonian import HubbardHamiltonian
from measure import do_measure
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
                f"t={system_cfg['t']}_"
                f"U={system_cfg['U']}_"
                f"Ne={sum(system_cfg['n_fermions_per_spin'])}"
            ),
            config=cfg,
        )



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

    
        last_time = now

        e = log_data["Energy"]


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
            }

            if hasattr(e, "variance"):
                payload["energy variance per site"] = float(e.variance.real / n_sites)

            wandb.log(payload)

        return True

    return wb_callback


def make_graph_and_hilbert(L: int, n_fermions_per_spin: tuple[int, int]):
    graph = nk.graph.Square(L)
    N = graph.n_nodes
    hi = nk.hilbert.SpinOrbitalFermions(
        N,
        s=1 / 2,
        n_fermions_per_spin=n_fermions_per_spin,
    )
    return graph, N, hi


def build_hubbard_hamiltonian(
    t: float,
    U: float,
    graph,
    hilbert,
):
    ham = 0.0

    for sz in (-1, 1):
        for u, v in graph.edges():
            ham += -t * cdag(hilbert, u, sz) @ c(hilbert, v, sz)
            ham += -t * cdag(hilbert, v, sz) @ c(hilbert, u, sz)

    for u in graph.nodes():
        ham += U * nc(hilbert, u, 1) @ nc(hilbert, u, -1)

    ham_jax = ham.to_jax_operator()

    return ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
        ham_jax
    )


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

    L = cfg["system"]["L"]
    t = cfg["system"]["t"]
    U = cfg["system"]["U"]
    n_fermions_per_spin = tuple(cfg["system"]["n_fermions_per_spin"])
    n_bands = cfg["system"]["n_bands"]
    seed = cfg["system"]["seed"]

    graph, N, hi = make_graph_and_hilbert(L, n_fermions_per_spin)
    Ne = sum(n_fermions_per_spin)

    key = jax.random.key(seed)
    trial_state = hi.random_state(key, size=1)

    if is_main_process():
        print("trial_state =", trial_state)
        print("*" * 80)
        print("*" * 20 + f" New calculation started. The local time is: {now} " + "*" * 20)
        print(f"System: {L}x{L} lattice with {N} sites")
        print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
        print(f"Parameters: t={t}, U={U}")
        print(f"n_bands={n_bands}")
        print()

    model = build_model(cfg, Ne=Ne, Ns=L * L)

    key, subkey = jax.random.split(key)
    params = model.init(subkey, trial_state)

    log_psi = model.apply(params, trial_state)
    if is_main_process():
        print(f"{log_psi.shape = }")

    # H = build_hubbard_hamiltonian(t=t, U=U, graph=graph, hilbert=hi)
    H = HubbardHamiltonian(t=t, U=U, graph=graph, hilbert=hi)

    sampler_cfg = cfg["sampling"]
    sampler = nk.sampler.MetropolisFermionHop(
        hi,
        graph=graph,
        d_max=sampler_cfg["d_max"],
        n_chains=sampler_cfg["n_chains"]
    )


    optimizer, lr_schedule = make_optimizer(cfg)
    print("sweep_size =", sampler.sweep_size)
    print("hi.size =", hi.size)

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

    
    shapes = jax.tree.map(lambda x: x.shape, vstate.parameters)
    

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
    callback = make_wb_callback(N, lr_schedule, cfg)

    vmc.run(
        n_iter=n_iter,
        out=log,
        callback=callback
    )



    energy_history = np.asarray(log.data["Energy"]["Mean"].real)
    full_energy = energy_history[-1]
    runtime = time.time() - total_start_time

    model_cfg = cfg["model"]

    embed_dim = model_cfg["embed_dim"]
    n_heads = model_cfg["n_heads"]
    num_layers = model_cfg["num_layers"]

    if is_main_process():
        print("Run ended")
        print("saving the state")
        common_tag = (
            f"t={t:.3f}_U={U:.3f}_"
            f"L={L:d}_Ne={Ne:d}_iter={n_iter:d}_"
            f"embed={embed_dim:d}_heads={n_heads:d}_layers={num_layers:d}_seed={seed:d}"
        )

        config_filename = output_dir / f"config_{common_tag}.yaml"

        with open(config_filename, "w") as f:
            yaml.dump(cfg, f)

        # state_filename = output_dir / f"state_after_VMC_{common_tag}.mpack"
        #
        # with open(state_filename, "wb") as file:
        #     file.write(to_bytes(vstate))
        # print("state saved")


        print(f"Optimized energy : {full_energy}")
        print(f"number of parameters : {int(n_params)}")
        print("total time cost is", runtime)





    

    order = cfg.get("observables", {}).get("order", [])
    path_w = output_dir

    do_measure(
        order,
        vstate,
        H,
        path_w,
        U,
        L,
        Ne,
        n_iter,
        graph,
        hi,
        embed_dim,
        n_heads,
        num_layers,
        seed,
    )
    
    if cfg.get("wandb", {}).get("use", False) and is_main_process():
        wandb.finish()
       


if __name__ == "__main__":
    run()