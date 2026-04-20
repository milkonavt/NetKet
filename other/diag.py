import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

# 1. Define the 3x3 lattice with PBC
L = 3
N = L * L
edges = []

# Generate the 18 bonds
for x in range(L):
    for y in range(L):
        site = x * L + y
        right = x * L + (y + 1) % L           # Horizontal PBC
        down = ((x + 1) % L) * L + y          # Vertical PBC
        edges.append((site, right))
        edges.append((site, down))

print(f"Total sites: {N}")
print(f"Total bonds: {len(edges)}")

# 2. Build the Hilbert space (3^9 = 19683 states)
num_states = 3**N
powers_of_3 = 3**np.arange(N)

def get_colors(state):
    """Decode integer to a base-3 array representing flavors on the 9 sites."""
    return (state // powers_of_3) % 3

def get_state(colors):
    """Encode the base-3 flavor array back into a single integer state index."""
    return np.sum(colors * powers_of_3)

# 3. Construct the Hamiltonian Matrix
print("Constructing the sparse Hamiltonian matrix...")
H = lil_matrix((num_states, num_states), dtype=np.float64)

for s in range(num_states):
    colors = get_colors(s)
    
    for i, j in edges:
        if colors[i] == colors[j]:
            # Same flavor: P_ij leaves state unchanged (+1 on diagonal)
            H[s, s] += 1.0
        else:
            # Different flavor: P_ij swaps them (+1 on off-diagonal)
            new_colors = colors.copy()
            new_colors[i], new_colors[j] = new_colors[j], new_colors[i]
            s_prime = get_state(new_colors)
            H[s, s_prime] += 1.0

# Convert to Compressed Sparse Row (CSR) format for fast eigenvalue solving
H_csr = H.tocsr()

# 4. Solve for the lowest eigenvalue
print("Diagonalizing...")
# k=1 (lowest eigenpair), which='SA' (Smallest Algebraic eigenvalue)
eigenvalues, eigenvectors = eigsh(H_csr, k=1, which='SA', tol=1e-8)

print(f"\nNumerical Ground State Energy: {eigenvalues[0]:.6f}")