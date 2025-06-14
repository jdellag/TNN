from copy import deepcopy
from dataclasses import dataclass
import numba
import numpy as np
import torch

from etnn import utils
from etnn.utils import scatter_mean, scatter_max
def compute_invariants(
    pos: torch.Tensor,
    frac_pos: torch.Tensor,
    lattice: torch.Tensor,
    cell_ind: dict[str, list[list[int]]],
    adj: dict[str, torch.LongTensor],
    hausdorff: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Compute PBC‐aware geometric invariants for each cell adjacency.

    Parameters
    ----------
    pos : torch.Tensor [N×3]
        Cartesian node coordinates (unused for PBC, kept for compatibility).
    frac_pos : torch.Tensor [N×3]
        Fractional node coordinates in [0,1)^3.
    lattice : torch.Tensor [3×3]
        Cell basis matrix whose columns are the primitive vectors.
    cell_ind : dict[str → list[list[int]]]
        For each rank (e.g. "0","1",...), a list of cells, each cell given by its node‐indices.
    adj : dict[str → LongTensor of shape 2×E]
        Adjacency between cell ranks: keys like "0_1", values are two rows of sender/receiver cell indices.
    hausdorff : bool
        If True, append a PBC Hausdorff distance feature.

    Returns
    -------
    features : dict[str → Tensor of shape E×F]
        For each adjacency rank_pair, an (E×F) tensor of invariants:
          F = 3 (centroid‐dist, sender‐diameter, receiver‐diameter) [+1 if hausdorff].
    """
    device = frac_pos.device
    # 1) Precompute centroids and diameters for each rank
    centroids: dict[str, torch.Tensor] = {}
    diameters: dict[str, torch.Tensor] = {}

    for rank, cells in cell_ind.items():
        # Flatten all node‐indices for this rank
        ids, index = [], []
        for i, cell_nodes in enumerate(cells):
            ids.extend(cell_nodes)
            index.extend([i] * len(cell_nodes))

        if len(ids) == 0:
            # No cells: empty tensors
            centroids[rank] = frac_pos.new_empty((0, 3))
            diameters[rank] = frac_pos.new_empty((0,))
            continue

        ids_tensor    = torch.tensor(ids,    dtype=torch.long, device=device)
        index_tensor  = torch.tensor(index,  dtype=torch.long, device=device)

        # 1a) Fractional centroids → Cartesian
        frac_cent = scatter_mean(frac_pos[ids_tensor], index_tensor, dim=0,
                                 dim_size=len(cells))        # [num_cells×3]
        centroids[rank] = frac_cent @ lattice.T               # [num_cells×3]

        # 1b) PBC‐aware diameters (max distance within each cell)
        # Compute all pairwise distances within each cell
        # We reuse the scatter_max on the flattened list of all pairs
        pair_dists = []
        pair_idx   = []
        for i, cell_nodes in enumerate(cells):
            for u in cell_nodes:
                for v in cell_nodes:
                    # minimum‐image in fractional space
                    df = frac_pos[v] - frac_pos[u]
                    df_wrap = df - torch.round(df)
                    dc = df_wrap @ lattice.T
                    pair_dists.append(dc.norm())
                    pair_idx.append(i)
        pair_dists_tensor = torch.stack(pair_dists)                       # [total_pairs]
        pair_idx_tensor   = torch.tensor(pair_idx, dtype=torch.long,
                                         device=device)                  # [total_pairs]
        diameters[rank] = scatter_max(pair_dists_tensor, pair_idx_tensor,
                                      dim=0, dim_size=len(cells))      # [num_cells]

    # 2) Build per‐adjacency features
    features: Dict[str, torch.Tensor] = {}
    for rank_pair, cell_pairs in adj.items():
        send_rank, recv_rank = rank_pair.split("_")
        send_idx = cell_pairs[0]  # shape [E]
        recv_idx = cell_pairs[1]  # shape [E]

        # 2a) Centroid–centroid distance
        cs = centroids[send_rank][send_idx]  # [E×3]
        cr = centroids[recv_rank][recv_idx]  # [E×3]
        centroid_dist = torch.norm(cs - cr, dim=1)  # [E]

        # 2b) Sender/receiver diameters
        d_send = diameters[send_rank][send_idx]    # [E]
        d_recv = diameters[recv_rank][recv_idx]    # [E]

        # 2c) Stack into [E×3]
        feats = torch.stack([centroid_dist, d_send, d_recv], dim=1)

        # 2d) Optional Hausdorff distance under PBC
        if hausdorff:
            from etnn.invariants import compute_hausdorff_distances
            # For each pair of cells, compute max_{u in S, v in R} ||u-v|| under PBC
            haus_vals = []
            for u_idx, v_idx in zip(send_idx.tolist(), recv_idx.tolist()):
                S = cell_ind[send_rank][u_idx]
                R = cell_ind[recv_rank][v_idx]
                max_dist = 0.0
                for u in S:
                    for v in R:
                        df = frac_pos[v] - frac_pos[u]
                        df_wrap = df - torch.round(df)
                        dc = df_wrap @ lattice.T
                        max_dist = max(max_dist, dc.norm().item())
                haus_vals.append(max_dist)
            haus_tensor = torch.tensor(haus_vals, device=device).unsqueeze(1)  # [E×1]
            feats = torch.cat([feats, haus_tensor], dim=1)                    # [E×4]

        features[rank_pair] = feats

    return features

# compute_invariants.num_features_map = defaultdict(lambda: 5)  # not needed, better to compute dynamic


def compute_max_pairwise_distances(cells, pos):
    """
    Compute each cell’s diameter (max internal pairwise distance).

    Parameters
    ----------
    cells : list[list[int]]
        Each inner list is the node‐indices for one cell.
    pos : torch.Tensor, shape [N, D]
        Cartesian coordinates of all N nodes.

    Returns
    -------
    torch.Tensor, shape [len(cells)]
        diameter[i] = max_{u,v in cells[i]} ||pos[u] - pos[v]||
    """
    # Use intercell‐distance routine with sender=receiver=cells
    dist_mat = compute_intercell_distances(cells, cells, pos)
    # The diameter of cell i is the (i,i) entry
    return torch.diagonal(dist_mat)

def compute_hausdorff_distances(
    sender_cells: torch.FloatTensor,
    receiver_cells: torch.FloatTensor,
    pos: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Compute the Hausdorff distances between two sets of cells.

    The Hausdorff distance is calculated based on the positions of sender and receiver cells. For
    two cells A and B where A is the sender and B is the receiver, the two Hausdorff distances
    computed by this function correspond to the maximum distance between any node in cell A and the
    node in cell B that is closest to it, and vice versa.

    Parameters
    ----------
    sender_cells : torch.FloatTensor
        A tensor representing the positions of the sender cells.
    receiver_cells : torch.FloatTensor
        A tensor representing the positions of the receiver cells.
    pos : torch.FloatTensor
        A tensor representing additional position information used in computing intercell distances.

    Returns
    -------
    torch.FloatTensor
        A tensor of shape (N, 2), where N is the number of sender cells. Each element contains the
        Hausdorff distance from sender to receiver cells in the first column, and receiver to sender
        cells in the second column.
    """
    dist_matrix = compute_intercell_distances(sender_cells, receiver_cells, pos)
    dist_matrix = dist_matrix.nan_to_num(float("inf"))

    sender_mins = dist_matrix.min(dim=2)[0]
    # Cast inf to -inf to correctly compute maxima
    sender_mins = torch.where(sender_mins == float("inf"), float("-inf"), sender_mins)
    sender_hausdorff = sender_mins.max(dim=1)[0]

    receiver_mins = dist_matrix.min(dim=1)[0]
    # Cast inf to -inf to correctly compute maxima
    receiver_mins = torch.where(
        receiver_mins == float("inf"), float("-inf"), receiver_mins
    )
    receiver_hausdorff = receiver_mins.max(dim=1)[0]

    hausdorff_distances = torch.stack([sender_hausdorff, receiver_hausdorff], dim=1)

    return hausdorff_distances


def compute_intercell_distances(sender_cells, receiver_cells, pos):
    """
    Compute the maximum distance between every pair of cells.

    Parameters
    ----------
    sender_cells : list[list[int]]
        Each inner list is the node‐indices for one “sender” cell.
    receiver_cells : list[list[int]]
        Each inner list is the node‐indices for one “receiver” cell.
    pos : torch.Tensor, shape [N, D]
        Cartesian coordinates of all N nodes.

    Returns
    -------
    torch.Tensor, shape [len(sender_cells), len(receiver_cells)]
        distance_matrix[i, j] = max_{u in sender_cells[i], v in receiver_cells[j]} ||pos[u] - pos[v]||
    """
    num_s = len(sender_cells)
    num_r = len(receiver_cells)
    # Preallocate output
    distance_matrix = pos.new_zeros((num_s, num_r))

    for i, s in enumerate(sender_cells):
        for j, r in enumerate(receiver_cells):
            if len(s) == 0 or len(r) == 0:
                distance_matrix[i, j] = 0.0
                continue
            # Gather coordinates
            coords_s = pos[s]              # [|s|, D]
            coords_r = pos[r]              # [|r|, D]
            # Compute all pairwise distances
            # coords_s[:, None, :] expands to [|s|,1,D]
            # coords_r[None, :, :] expands to [1,|r|,D]
            diffs = coords_s[:, None, :] - coords_r[None, :, :]  # [|s|,|r|,D]
            dists = torch.norm(diffs, dim=-1)                    # [|s|,|r|]
            # Maximum over all pairs
            distance_matrix[i, j] = dists.max()
    return distance_matrix


def compute_centroids(cells, pos):
    """
    Compute Cartesian centroids for a list of combinatorial cells.

    Parameters
    ----------
    cells : list[list[int]]
        Each entry is a cell, given by the list of node indices it contains.
    pos : torch.Tensor, shape [N, D]
        Cartesian coordinates of the N nodes.

    Returns
    -------
    torch.Tensor, shape [num_cells, D]
        The centroid of each cell, computed by averaging its node positions.
    """
    import torch
    from etnn.utils import scatter_mean

    # 1) Flatten all node indices and remember which cell they came from
    ids = []
    idx_map = []
    for cell_id, cell_nodes in enumerate(cells):
        ids.extend(cell_nodes)
        idx_map.extend([cell_id] * len(cell_nodes))

    # 2) Handle the empty‐cells case
    if len(ids) == 0:
        # No cells → return empty tensor with correct number of dims
        return pos.new_empty((0, pos.size(1)))

    # 3) Build tensors for scatter_mean
    ids_tensor    = torch.tensor(ids,    dtype=torch.long, device=pos.device)
    idx_map_tensor = torch.tensor(idx_map, dtype=torch.long, device=pos.device)

    # 4) Compute centroids by averaging node positions per cell
    centroids = scatter_mean(
        pos[ids_tensor],      # [total_nodes_in_all_cells, D]
        idx_map_tensor,       # [total_nodes_in_all_cells]
        dim=0,
        dim_size=len(cells)   # number of cells
    )
    return centroids

@dataclass
class SparseInvariantComputationIndices:
    """This auxiliary class is used to compute the indices for the computation of geometric
    invariants. The agents permit to compute centroids, diameters, and Hausdorff distances between
    pairs of cells using only scatter add/min/max/mean operations in linear memory."""

    cell_ids: np.ndarray
    atoms_ids_send: np.ndarray
    atoms_ids_recv: np.ndarray
    haus_min_recv: np.ndarray
    haus_min_send: np.ndarray
    haus_max_recv: np.ndarray
    haus_max_send: np.ndarray


@numba.jit(nopython=True)
def _sparse_computation_indices(
    atoms_send: np.ndarray,
    slices_send: np.ndarray,
    atoms_recv: np.ndarray,
    slices_recv: np.ndarray,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    # inputs must be equal size
    splits_send = np.split(atoms_send, np.cumsum(slices_send)[:-1])
    splits_recv = np.split(atoms_recv, np.cumsum(slices_recv)[:-1])

    # must return a tuple of 6 arrays
    n = len(slices_send)  # must be the same as len(splits_recv)
    m = sum(slices_send * slices_recv)
    ids_send = np.empty(m, dtype=np.int64)
    ids_recv = np.empty(m, dtype=np.int64)
    minindex_send = np.empty(m, dtype=np.int64)
    minindex_recv = np.empty(m, dtype=np.int64)
    cell_ids = np.empty(m, dtype=np.int64)

    offset_index_send = 0
    offset_index_recv = 0
    step = 0
    for i in range(n):
        n_send = len(splits_send[i])
        n_recv = len(splits_recv[i])
        for j, sj in enumerate(splits_send[i]):
            for k, sk in enumerate(splits_recv[i]):
                ind = step + j * n_recv + k
                ids_send[ind] = sj
                ids_recv[ind] = sk
                minindex_send[ind] = offset_index_send + j
                minindex_recv[ind] = offset_index_recv + k
                cell_ids[ind] = i
        step += n_send * n_recv
        offset_index_send += n_send
        offset_index_recv += n_recv

    step = 0
    maxindex_send = np.empty(sum(slices_send), dtype=np.int64)
    for j, sj in enumerate(splits_send):
        maxindex_send[step : step + len(sj)] = j
        step += len(sj)

    step = 0
    maxindex_recv = np.empty(sum(slices_recv), dtype=np.int64)
    for j, sj in enumerate(splits_recv):
        maxindex_recv[step : step + len(sj)] = j
        step += len(sj)

    return (
        cell_ids,
        ids_send,
        ids_recv,
        minindex_send,
        minindex_recv,
        maxindex_send,
        maxindex_recv,
    )


def sparse_computation_indices_from_cc(cell_ind, adj, max_cell_size=100):
    agg_indices = {}

    # first take sub sample of the cells if larger than max size
    cell_ind = deepcopy(cell_ind)
    for rank, cells in cell_ind.items():
        for j, c in enumerate(cells):
            if len(c) > max_cell_size:
                subsample = np.random.choice(c, max_cell_size, replace=False)
                cell_ind[rank][j] = subsample.tolist()

        # create agg indices for rank
        # transform on the format of atoms/sizes needed by the helper functin
        atoms = np.concatenate(cells)
        slices = np.array([len(c) for c in cells])
        indices = _sparse_computation_indices(atoms, slices, atoms, slices)
        indices = SparseInvariantComputationIndices(*indices)
        agg_indices[rank] = indices
        # [x.tolist() for x in indices]

    # for each aggregation indices for edges
    for adj_type, edges in adj.items():
        # get the indices of the cells
        send_rank, recv_rank = adj_type.split("_")[:2]

        # get the cells of sender and receiver
        cells_send = [cell_ind[send_rank][i] for i in edges[0]]
        cells_recv = [cell_ind[recv_rank][i] for i in edges[1]]

        # transform on the format of atoms/sizes needed by the helper functin
        atoms_send = np.concatenate(cells_send)
        slices_send = np.array([len(c) for c in cells_send])
        atoms_recv = np.concatenate(cells_recv)
        slices_recv = np.array([len(c) for c in cells_recv])

        # get the indices of the cells
        indices = _sparse_computation_indices(
            atoms_send, slices_send, atoms_recv, slices_recv
        )
        indices = SparseInvariantComputationIndices(*indices)
        agg_indices[adj_type] = indices

    return agg_indices, cell_ind


def compute_invariants_sparse(
    pos: torch.FloatTensor,
    cell_ind: dict[str, list[list[int]]],
    adj: dict[str, torch.LongTensor],
    rank_agg_indices: dict[str, SparseInvariantComputationIndices],
    hausdorff: bool = True,
    diff_high_order: bool = False,
) -> dict[str, torch.Tensor]:
    """This function computes the geometric invariants between pairs of cells specified in `adj`

    Parameters
    ----------
    pos : torch.FloatTensor
        A 2D tensor of shape (num_nodes, num_dimensions) containing the positions of each node.
    cell_ind : dict
        A dictionary mapping cell ranks to lists of lists of node indices for each cell.
    adj : dict
        A dictionary where each key is a string in the format 'sender_rank_receiver_rank' indicating
        the ranks of cell pairs, and each value is a tensor of shape (2, num_cell_pairs) containing
        indices for sender and receiver cells.
    rank_agg_indices : dict
        A dictionary where each key is a rank and each value is a SparseInvariantComputationIndices
        object that contains the indices for the computation of geometric invariants.
    hausdorff : bool
        Whether to compute the Hausdorff distances between cells. Default is True.
    diff_high_order : bool
        Whether to compute the distances between high-order nodes. Default is False.
    """
    # device
    dev = pos.device

    # placeholder
    inv_list: dict[str, list[torch.Tensor]] = {}

    # compute centroids and diameters
    centroids: dict[str, torch.Tensor] = {}
    diameters: dict[str, torch.Tensor] = {}
    for rank, cells in cell_ind.items():
        # gather node indices flatten for this rank
        ids: list[int] = []
        for c in cells:
            ids.extend(c)
        # build a scatter-index mapping each node to its cell sizes
        sizes, index = [], []
        for i, c in enumerate(cells):
            sizes.append(len(c)); index.extend([i] * len(c))
        index = torch.tensor(index, device=dev)
        # compute fractional centroids under PBC
        frac_centroid = utlis.scatter_mean(
                frac_pos[ids], index, dim = 0, dim_size = len(cells)
                )
        # map back to cartesian coordinates
        centroids[rank] = frac_centroid @ lattice.T

        # compute diameters (max pairwise distance)
        agg = rank_agg_indices[rank]
        # PBC‐aware diameter: wrap each node‐pair in fractional space
        fp_send = frac_pos[agg.atoms_ids_send]
        fp_recv = frac_pos[agg.atoms_ids_recv]
        df      = fp_recv - fp_send
        df_wrap = df - torch.round(df)                # minimum‐image frac
        dc      = df_wrap @ lattice.T                 # back to Cartesian
        dist    = torch.norm(dc, dim=-1)              # [#pairs]
        index = torch.tensor(agg.cell_ids).to(dev)
        diameters[rank] = utils.scatter_max(dist, index, dim=0, dim_size=len(cells))

    # compute distances
    for rank_pair, cell_pairs in adj.items():
        send_rank, recv_rank = rank_pair.split("_")

        # check if reverse message has been computed
        flipped_rank_pair = f"{recv_rank}_{send_rank}"
        if flipped_rank_pair in inv_list:
            inv_list[rank_pair] = inv_list[flipped_rank_pair]
            continue

        # centroid dist
        # PBC‐aware centroid distance
        c_send = centroids[send_rank][cell_pairs[0]]
        c_recv = centroids[recv_rank][cell_pairs[1]]
        # but centroids were already Cartesian‐mapped, so simple norm
        centroids_dist = torch.norm(c_send - c_recv, dim=1)

        # diameter
        diameter_send = diameters[send_rank][cell_pairs[0]]
        diameter_recv = diameters[recv_rank][cell_pairs[1]]

        inv_list[rank_pair] = [
            centroids_dist,
            diameter_send,
            diameter_recv,
        ]

        # hausdorff
        if hausdorff:
            # gather indices for positions
            agg = rank_agg_indices[rank_pair]
            if diff_high_order:
                pos_send = pos[agg.atoms_ids_send]
                pos_recv = pos[agg.atoms_ids_recv]
            else:
                pos_send = pos.detach()[agg.atoms_ids_send]
                pos_recv = pos.detach()[agg.atoms_ids_recv]
            dists = torch.norm(pos_send - pos_recv, dim=1)

            # move agg indices to device for scatter operations
            minix_send = torch.tensor(agg.haus_min_send).to(dev)
            minix_recv = torch.tensor(agg.haus_min_recv).to(dev)
            maxix_send = torch.tensor(agg.haus_max_send).to(dev)
            maxix_recv = torch.tensor(agg.haus_max_recv).to(dev)

            # compute hausdorff distances with scatter operations
            hausdorff_send = utils.scatter_max(
                utils.scatter_min(dists, minix_send), maxix_send
            )
            hausdorff_recv = utils.scatter_max(
                utils.scatter_min(dists, minix_recv), maxix_recv
            )

            inv_list[rank_pair].extend([hausdorff_send, hausdorff_recv])

    out: dict[str, torch.Tensor] = {
        k: torch.stack(v, dim=1) for k, v in inv_list.items()
    }

    return out
