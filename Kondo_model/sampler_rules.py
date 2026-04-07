import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from flax import struct



class KondoFlipRule(nk.sampler.rules.MetropolisRule):
    n_sites: int = struct.field(pytree_node=False)

    def __init__(self, n_sites: int):
        self.n_sites = n_sites


    def transition(self, sampler, machine, params, sampler_state, key, sigma):
        """
        sigma layout:
            sigma[:, 0:N]     = electron down occupations
            sigma[:, N:2*N]   = electron up occupations
            sigma[:, 2*N:3*N] = local spins (+1 / -1)

        Allowed flips:
            (0,1,-1) -> (1,0,+1)   [S^+ s^-]
            (1,0,+1) -> (0,1,-1)   [S^- s^+]
        """
        batch = sigma.shape[0]
        N = self.n_sites
        row = jnp.arange(batch)

        dn = sigma[:, 0:N]
        up = sigma[:, N:2 * N]
        sp = sigma[:, 2 * N:3 * N]

        # Flippable sites:
        #   electron up + local spin down
        #   electron down + local spin up
        can_plus_site = (dn == 0) & (up == 1) & (sp == -1)
        can_minus_site = (dn == 1) & (up == 0) & (sp == +1)
        flippable = can_plus_site | can_minus_site

        # Some chains may have no valid site; those chains will remain unchanged.
        has_move = jnp.any(flippable, axis=1)

        # Sample one valid site per chain.
        # For invalid sites, give a very negative logit.
        logits = jnp.where(flippable, 0.0, -1.0e9)
        i = jax.random.categorical(key, logits=logits, axis=1)

        dn_i = dn[row, i]
        up_i = up[row, i]
        sp_i = sp[row, i]

        can_plus = has_move & (dn_i == 0) & (up_i == 1) & (sp_i == -1)
        can_minus = has_move & (dn_i == 1) & (up_i == 0) & (sp_i == +1)

        sigma_p = sigma

        # (0,1,-1) -> (1,0,+1)
        sigma_p = sigma_p.at[row, i].set(
            jnp.where(can_plus, 1, sigma_p[row, i])
        )
        sigma_p = sigma_p.at[row, N + i].set(
            jnp.where(can_plus, 0, sigma_p[row, N + i])
        )
        sigma_p = sigma_p.at[row, 2 * N + i].set(
            jnp.where(can_plus, +1, sigma_p[row, 2 * N + i])
        )

        # (1,0,+1) -> (0,1,-1)
        sigma_p = sigma_p.at[row, i].set(
            jnp.where(can_minus, 0, sigma_p[row, i])
        )
        sigma_p = sigma_p.at[row, N + i].set(
            jnp.where(can_minus, 1, sigma_p[row, N + i])
        )
        sigma_p = sigma_p.at[row, 2 * N + i].set(
            jnp.where(can_minus, -1, sigma_p[row, 2 * N + i])
        )

        return sigma_p, None
class IdentityRule(nk.sampler.rules.MetropolisRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        return sigma, None

def make_sz0_initial_states(N, n_fermions, n_chains, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros((n_chains, 3 * N), dtype=np.int8)

    for b in range(n_chains):
        while True:
            # choose N_up, N_down with fixed total fermion number
            n_up = rng.integers(0, n_fermions + 1)
            n_down = n_fermions - n_up

            # need spin sum to cancel fermion spin imbalance:
            # 0.5*(n_up-n_down) + 0.5*sum(spin) = 0
            # so sum(spin) = -(n_up - n_down)
            target_spin_sum = -(n_up - n_down)

            # feasible only if within range and parity works
            if abs(target_spin_sum) > N:
                continue
            if (N + target_spin_sum) % 2 != 0:
                continue

            up = np.zeros(N, dtype=np.int8)
            dn = np.zeros(N, dtype=np.int8)

            if n_up > 0:
                up_sites = rng.choice(N, size=n_up, replace=False)
                up[up_sites] = 1

            if n_down > 0:
                dn_sites = rng.choice(N, size=n_down, replace=False)
                dn[dn_sites] = 1

            # number of +1 spins needed
            n_spin_up = (N + target_spin_sum) // 2
            sp = -np.ones(N, dtype=np.int8)
            if n_spin_up > 0:
                sp_sites = rng.choice(N, size=n_spin_up, replace=False)
                sp[sp_sites] = +1

            states[b] = np.concatenate([dn,up, sp])
            break

    return jnp.array(states)