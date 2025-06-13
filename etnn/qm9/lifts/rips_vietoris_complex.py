# etnn/qm9/lifts/rips_vietoris_complex.py

from itertools import combinations
import torch
import gudhi
from torch_geometric.data import Data

from etnn.combinatorial_data import Cell

# Number of dummy features attached to each simplex
NUM_FEATURES = 0


def rips_lift(graph: Data, dim: int, dis: float, fc_nodes: bool = True) -> set[Cell]:
    """
    Construct a Rips complex from a graph (with periodic boundary conditions)
    and return its simplices as a set of Cells.

    Parameters
    ----------
    graph : Data
        A PyG Data object containing:
          - x:         node feature tensor [N×F]
          - frac_pos:  fractional node coords [N×3] in [0,1)^3
          - lattice:   3×3 cell basis matrix
    dim : int
        Maximum simplex dimension (i.e. build up to dim‐simplices).
    dis : float
        Cutoff distance (in the same units as lattice·frac_pos).
    fc_nodes : bool, optional
        If True, ensure every 1‐simplex between node pairs is included.

    Returns
    -------
    set[Cell]
        A set of Cells, where each Cell is (frozenset(node_indices), feature_tuple).
    """
    # Unpack graph attributes
    x_0      = graph.x
    pos_frac = graph.frac_pos    # [N, 3]
    lattice  = graph.lattice     # [3, 3]

    # ——— PBC neighbor‐listing (matching vendored signature) ———
    # 1) Cartesian coords from fractional
    cart_coords = pos_frac @ lattice.T                 # [N×3]

    # 2) Cell lengths & angles
    a, b, c = lattice[:, 0], lattice[:, 1], lattice[:, 2]
    lengths = torch.stack([a.norm(), b.norm(), c.norm()])  # [3]
    alpha = torch.acos((b @ c) / (b.norm() * c.norm()))
    beta  = torch.acos((a @ c) / (a.norm() * c.norm()))
    gamma = torch.acos((a @ b) / (a.norm() * b.norm()))
    angles = torch.stack([alpha, beta, gamma])            # [3]

    # 3) Call vendored PBC routine
    num_atoms = torch.tensor([cart_coords.size(0)],
                         dtype=torch.long,
                         device=cart_coords.device)
    device    = cart_coords.device
        # ——— PBC neighbor‐listing (manual, minimal) ———
    N = pos_frac.size(0)
    edge_list = []
    dist_list = []
    offset_list = []

    # for every unordered pair i<j compute the wrapped distance
    for i in range(N):
        for j in range(i+1, N):
            df = pos_frac[j] - pos_frac[i]                           # Δ frac
            df_wrap = df - torch.round(df)                           # minimum‐image frac
            dc = (df_wrap @ lattice.T)                               # back to Cartesian
            d = dc.norm().item()
            if d <= dis:
                # record both directions
                off = (-torch.round(df)).long()                     # offset so frac+off=df_wrap
                edge_list += [[i, j], [j, i]]
                dist_list += [d, d]
                offset_list += [off.tolist(), (-off).tolist()]

    # pack into tensors (or empty tensors if no edges)
    if edge_list:
        edge_index  = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2×E]
        distances   = torch.tensor(dist_list,  dtype=torch.float)                # [E]
        cell_offsets= torch.tensor(offset_list, dtype=torch.long)                # [E×3]
    else:
        edge_index   = torch.empty((2, 0), dtype=torch.long)
        distances    = torch.empty((0,),    dtype=torch.float)
        cell_offsets = torch.empty((0, 3),  dtype=torch.long)
    # ————————————————————————————————————————————————
    # Build the simplex tree from these edges
    simplex_tree = gudhi.SimplexTree()
    # Insert all 0‐simplices (nodes)
    for idx in range(x_0.size(0)):
        simplex_tree.insert([idx])

    # Insert 1‐simplices (PBC edges)
    for (u, v), d in zip(edge_index.t().tolist(), distances.tolist()):
        # We know d ≤ dis already
        simplex_tree.insert([u, v])

    # Optionally force‐connect every node pair as an edge
    if fc_nodes:
        nodes = list(range(x_0.size(0)))
        for u, v in combinations(nodes, 2):
            simplex_tree.insert([u, v])

    # At this point, you may wish to manually expand the tree
    # to higher‐order cells (2‐simplices, 3‐simplices, …) up to `dim`,
    # since we bypassed `RipsComplex.create_simplex_tree`. For example:
    # simplex_tree.expand_to_dimension(dim)

    # Convert to a set of frozensets (Cells)
    simplexes = set()
    for simplex, _ in simplex_tree.get_simplices():
        # GUDHI returns all simplices in the tree (0 up to the inserted max)
        if len(simplex) - 1 <= dim:  # filter by dimension
            simplexes.add(frozenset(simplex))

    # Attach dummy feature vectors to each simplex
    dummy_features = tuple(range(NUM_FEATURES))
    simplexes = {(s, dummy_features) for s in simplexes}

    return simplexes


# Tell the framework how many features this lifter produces per simplex
rips_lift.num_features = NUM_FEATURES
