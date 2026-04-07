import yaml
import jax.numpy as jnp

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cluster = cfg["paths"]["cluster"]
if cluster is True:
    path_w = cfg["paths"]["measurement_cluster"]
    path_d = cfg["paths"]["data_cluster"]
else:
    path_w = cfg["paths"]["measurement_local"]
    path_d = cfg["paths"]["data_local"]

Lx = cfg["system"]["Lx"]
Ly = cfg["system"]["Ly"]
Ne = cfg["system"]["Ne"]
n_bands = cfg["system"]["n_bands"]
pbc = cfg["system"]["pbc"]

t = cfg["hamiltonian"]["t"]
U = cfg["hamiltonian"]["U"]
Jk = cfg["hamiltonian"]["Jk"]
J = cfg["hamiltonian"]["J"]

n_heads = cfg["model"]["n_heads"]
d_model = cfg["model"]["d_model"]
num_layers = cfg["model"]["num_layers"]
patch_size = cfg["model"]["patch_size"]
transl_invariant = cfg["model"]["transl_invariant"]

N_samples = cfg["sampler"]["n_samples"]
n_chains = cfg["sampler"]["n_chains"]
n_discard_per_chain = cfg["sampler"]["n_discard_per_chain"]
chunk_size = cfg["sampler"]["chunk_size"]
d_max = cfg["sampler"]["d_max"]
rule_probabilities = jnp.array(cfg["sampler"]["rule_probabilities"])

learning_rate = cfg["optimizer"]["learning_rate"]
diag_shift = cfg["optimizer"]["diag_shift"]
use_ntk = cfg["optimizer"]["use_ntk"]
mode = cfg["optimizer"]["mode"]
on_the_fly = cfg["optimizer"]["on_the_fly"]

seed = cfg["training"]["seed"]
init_seed = cfg["training"]["init_seed"]
N_opt = cfg["training"]["n_iter"]

wandb_project = cfg["wandb"]["project"]
use_wandb = cfg["wandb"]["use_wandb"]