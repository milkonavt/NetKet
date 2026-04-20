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

from Embedding import ViT


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def is_main_process():
    return jax.process_index() == 0


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(__file__).resolve().parent / path
    with open(config_path, "r", encoding="utf-8") as f:
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
                f"SU{system_cfg['N_color']}_"
                f"L={system_cfg['L']}_"
                f"J={system_cfg['J']}"
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

    def wb_callback(step, log_data, driver=None):
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


# ------------------------------------------------------------
# Square lattice + fundamental local Hilbert space
# ------------------------------------------------------------

def make_square_graph(L: int, pbc=True):
    graph = nk.graph.Square(L, pbc=pbc)
    return graph, graph.n_nodes


def make_suN_hilbert(n_colors: int, n_sites: int):
    local_states = nk.utils.StaticRange(0, 1, n_colors)
    hi = nk.hilbert.HomogeneousHilbert(local_states, N=n_sites)
    return hi


def permutation_bond_matrix(n_colors: int, J: float = 1.0):
    d = n_colors
    P = np.zeros((d * d, d * d), dtype=np.float64)

    for a in range(d):
        for b in range(d):
            bra = a * d + b
            ket = b * d + a
            P[bra, ket] = 1.0

    return J * P


def build_suN_hamiltonian(J: float, graph, hilbert, n_colors: int):
    Pij = permutation_bond_matrix(n_colors, J=J)
    return nk.operator.GraphOperator(
        hilbert=hilbert,
        graph=graph,
        bond_ops=[Pij],
        dtype=np.float64,
    )


def build_model(cfg: dict, Ns: int) -> ViT:
    model_cfg = cfg["model"]
    system_cfg = cfg["system"]
    dtype = get_dtype(model_cfg.get("param_dtype", "float64"))

    return ViT(
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["embed_dim"],
        n_heads=model_cfg["n_heads"],
        patch_size=model_cfg["patch_size"],
        transl_invariant=model_cfg["transl_invariant"],
        Ns=Ns,
        n_bands=system_cfg["N_color"],
        param_dtype=dtype,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

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

    system_cfg = cfg["system"]
    L = system_cfg["L"]
    pbc = system_cfg.get("pbc", True)
    n_colors = system_cfg["N_color"]
    J = system_cfg["J"]
    seed = system_cfg["seed"]

    graph, N_sites = make_square_graph(L, pbc=pbc)
    hi = make_suN_hilbert(n_colors, N_sites)

    key = jax.random.key(seed)
    trial_state = hi.random_state(key, size=1)

    print("Initial trial state:", trial_state)

    if is_main_process():
        print("trial_state =", trial_state)
        print("*" * 80)
        print("*" * 20 + f" New calculation started. The local time is: {now} " + "*" * 20)
        print(f"System: {L}x{L} lattice with {N_sites} sites")
        print(f"Model: SU({n_colors})")
        print(f"Parameters: J={J}")
        print()

    model = build_model(cfg, Ns=N_sites)

    key, subkey = jax.random.split(key)
    params = model.init(subkey, trial_state)

    log_psi = model.apply(params, trial_state)
    if is_main_process():
        print(f"{log_psi.shape = }")

    H = build_suN_hamiltonian(J=J, graph=graph, hilbert=hi, n_colors=n_colors)

    # from scipy.sparse.linalg import eigsh
    # H=H.to_sparse()
    

    # eigenvalues, eigenvectors = eigsh(H, k=6, which='SA')
    # print("Exact ground state energy =", eigenvalues)


    sampler_cfg = cfg["sampling"]    

    sampler = nk.sampler.MetropolisExchange(
        hi,
        graph=graph,
        d_max=sampler_cfg["d_max"],
        n_chains=sampler_cfg["n_chains"],
    )

    optimizer, lr_schedule = make_optimizer(cfg)

    if is_main_process():
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
    


    def make_equal_color_initial_states(n_sites, n_colors, n_chains, seed=0):
        if n_sites % n_colors != 0:
            raise ValueError(
                f"Equal-flavor sector requires n_sites % n_colors == 0, "
                f"got n_sites={n_sites}, n_colors={n_colors}"
            )

        rng = np.random.default_rng(seed)
        states = np.zeros((n_chains, n_sites), dtype=np.int8)

        n_per_color = n_sites // n_colors
        base = np.repeat(np.arange(n_colors, dtype=np.int8), n_per_color)

        for b in range(n_chains):
            states[b] = rng.permutation(base)

        return jnp.array(states)
    

    sigma_ref = vstate.sampler_state.σ

    sigma0 = make_equal_color_initial_states(
    n_sites=N_sites,
    n_colors=n_colors,
    n_chains=sampler_cfg["n_chains"],
    seed=seed,
    )


    sigma0 = jax.device_put(sigma0, sigma_ref.sharding)

    vstate.sampler_state = vstate.sampler_state.replace(σ=sigma0)

    print(vstate.sampler_state.σ[0:5])

    
    


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
    callback = make_wb_callback(N_sites, lr_schedule, cfg)

    vmc.run(
        n_iter=n_iter,
        out=log,
        callback=callback,
    )

    energy_history = np.asarray(log.data["Energy"]["Mean"].real)
    full_energy = energy_history[-1]
    runtime = time.time() - total_start_time

    if is_main_process():
        print("Run ended")
        print("Saving output")

        common_tag = (
            f"SU{n_colors}_J={J:.3f}_"
            f"L={L:d}_iter={n_iter:d}_seed={seed:d}"
        )

        config_filename = output_dir / f"config_{common_tag}.yaml"
        with open(config_filename, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        energy_filename = output_dir / f"energy_{common_tag}.txt"
        np.savetxt(energy_filename, energy_history)

        print(f"Optimized energy : {full_energy}")
        print(f"Energy per site  : {full_energy / N_sites}")
        print(f"Number of parameters : {int(n_params)}")
        print("Total time cost is", runtime)

    if cfg.get("wandb", {}).get("use", False) and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    run()