import time
import numpy as np
import jax
from jax.experimental import multihost_utils as mhu

from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd


def is_main_process():
    return jax.process_index() == 0


def do_measure(
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
):
    file_tag = (
        f"U={U:.3f}_L={L:d}_Ne={Ne:d}_iter={n_iter:d}_"
        f"embed={embed_dim:d}_heads={n_heads:d}_layers={num_layers:d}_seed={seed:d}"
    )

    start = time.time()

    # Keep all ranks aligned before measurement starts.
    mhu.sync_global_devices("measure_start")

    if is_main_process():
        print("measurement started", flush=True)

    if "E" in order:
        E = vstate.expect(H)

        if is_main_process():
            print("E measurement completed.", flush=True)
            np.savetxt(
                path_w / f"energy_{file_tag}.txt",
                [float(E.mean.real)],
            )

    if "NN" in order:
        density = np.zeros((graph.n_nodes,), dtype=float)

        for i in graph.nodes():
            N_op = nc(hi, i, 1) + nc(hi, i, -1)
            N_op = N_op.to_jax_operator()
            N_op = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(N_op)
            density[i] = float(vstate.expect(N_op).mean.real)

        if is_main_process():
            np.savetxt(path_w / f"N_{file_tag}.txt", density)

        NN_full = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)
        NN_conn = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)

        for i in graph.nodes():
            for j in graph.nodes():
                NN_op = (nc(hi, i, 1) + nc(hi, i, -1)) @ (nc(hi, j, 1) + nc(hi, j, -1))
                NN_op = NN_op.to_jax_operator()
                NN_op = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(NN_op)
                NN_full[i, j] = float(vstate.expect(NN_op).mean.real)
                NN_conn[i, j] = NN_full[i, j] - density[i] * density[j]

        if is_main_process():
            print("NN measurement completed.", flush=True)
            np.savetxt(path_w / f"NN_full_{file_tag}.txt", NN_full)
            np.savetxt(path_w / f"NN_{file_tag}.txt", NN_conn)

    if "CC" in order:
        CC = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)

        for i in graph.nodes():
            for j in graph.nodes():
                CC_op = cdag(hi, i, 1) @ c(hi, j, 1)
                CC_op = CC_op.to_jax_operator()
                CC_op = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(CC_op)
                CC[i, j] = float(vstate.expect(CC_op).mean.real)

        if is_main_process():
            print("CC measurement completed.", flush=True)
            np.savetxt(path_w / f"CC_{file_tag}.txt", CC)

    if "SS" in order:
        sz = np.zeros((graph.n_nodes,), dtype=float)

        for i in graph.nodes():
            Sz_op = 0.5 * (nc(hi, i, 1) - nc(hi, i, -1))
            Sz_op = Sz_op.to_jax_operator()
            Sz_op = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(Sz_op)
            sz[i] = float(vstate.expect(Sz_op).mean.real)

        if is_main_process():
            np.savetxt(path_w / f"Sz_{file_tag}.txt", sz)

        SS_full = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)
        SS_conn = np.zeros((graph.n_nodes, graph.n_nodes), dtype=float)

        for i in graph.nodes():
            for j in graph.nodes():
                Sz_i = 0.5 * (nc(hi, i, 1) - nc(hi, i, -1))
                Sz_j = 0.5 * (nc(hi, j, 1) - nc(hi, j, -1))

                Sp_i = cdag(hi, i, 1) @ c(hi, i, -1)
                Sm_i = cdag(hi, i, -1) @ c(hi, i, 1)

                Sp_j = cdag(hi, j, 1) @ c(hi, j, -1)
                Sm_j = cdag(hi, j, -1) @ c(hi, j, 1)

                SS_op = (Sz_i @ Sz_j) + 0.5 * ((Sp_i @ Sm_j) + (Sm_i @ Sp_j))
                SS_op = SS_op.to_jax_operator()
                SS_op = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(SS_op)

                SS_full[i, j] = float(vstate.expect(SS_op).mean.real)
                SS_conn[i, j] = SS_full[i, j] - sz[i] * sz[j]

        if is_main_process():
            print("Full spin correlator SS measurement completed.", flush=True)
            np.savetxt(path_w / f"SS_full_{file_tag}.txt", SS_full)
            np.savetxt(path_w / f"SS_{file_tag}.txt", SS_conn)

    # Keep all ranks aligned before returning to the caller.
    mhu.sync_global_devices("measure_end")

    end = time.time()
    if is_main_process():
        print(f"Execution time: {end - start:.3f} seconds", flush=True)