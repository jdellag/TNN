from copy import deepcopy
from dataclasses import dataclass
import numba
import numpy as np
import torch
from torch import Tensor
import math

from etnn import utils
from etnn.utils import scatter_mean, scatter_max
def compute_invariants(
    pos: Tensor,
    frac_pos: Tensor,
    lattice: Tensor,
    cell_ind: dict[str, list[list[int]]],
    adj: dict[str, Tensor],
    hausdorff: bool = True,
) -> dict[str, Tensor]:
    """
    Compute PBC‐aware geometric invariants for each cell adjacency.

    Parameters
    ----------
    pos : Tensor [N×3]
        Cartesian node coordinates (unused for PBC but kept for compatibility).
    frac_pos : Tensor [N×3]
        Fractional node coordinates in [0,1)^3.
    lattice : Tensor [3×3]
        Cell basis matrix whose columns are the primitive vectors.
    cell_ind : dict[str → list of list of int]
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
    print("\n===  BEGIN compute_invariants  ===")
    print(f"[DBG-0] frac_pos.shape = {frac_pos.shape}  lattice.shape = {lattice.shape}")
    # -------------------------------------------------------------------------
    # 1) Pre-compute centroids & diameters per rank
    centroids: dict[str, Tensor] = {}
    diameters: dict[str, Tensor] = {}

    for rank, cells in cell_ind.items():
        print(f"\n[DBG-1] Processing rank {rank}  (#cells = {len(cells)})")
        num_cells = len(cells)
        if num_cells == 0:
            centroids[rank] = torch.empty((0, 3), device=device, dtype=frac_pos.dtype)
            diameters[rank] = torch.empty((0,),  device=device, dtype=frac_pos.dtype)
            continue

        # 1.1)  Flatten node ids and build map → cell
        node_ids, cell_map = [], []
        for ci, cell_nodes in enumerate(cells):
            raw = cell_nodes.tolist() if isinstance(cell_nodes, torch.Tensor) else cell_nodes
            for n in raw:
                if isinstance(n, float) and math.isnan(n):
                    continue
                node_ids.append(int(n))
                cell_map.append(ci)
        print(f"[DBG-1a]   first 10 node_ids = {node_ids[:10]}")
        print(f"[DBG-1b]   min/max node_id  = {min(node_ids)} / {max(node_ids)}")

        node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=device)
        cell_map_t = torch.tensor(cell_map, dtype=torch.long, device=device)
        # ────────────────────────────────────────────────────────────────
        # 1.2)  Compute centroids in Cartesian coordinates
        #   ▸ crystals / periodic cells → lattice is [3,3]
        #   ▸ molecules (QM9)           → lattice has an extra dim or is dummy
        # ────────────────────────────────────────────────────────────────
        if lattice.dim() == 2:                                  # periodic
            frac_cent = scatter_mean(
                frac_pos[node_ids_t],                           # fractional
                cell_map_t,
                dim=0,
                dim_size=num_cells
            )                                                   # [num_cells,3]
            centroids[rank] = frac_cent @ lattice.T             # Cartesian
        else:                                                   # molecular
            centroids[rank] = scatter_mean(
                pos[node_ids_t],                                # already Cartesian
                cell_map_t,
                dim=0,
                dim_size=num_cells
            )                                                   # [num_cells,3]
        # ────────────────────────────────────────────────────────────────
        print(f"[DBG-1c]   centroids[{rank}].shape = {centroids[rank].shape}")

        # 1.3)  Diameters
        dia_vals: list[float] = []
        for cell_nodes in cells:
            raw = cell_nodes.tolist() if isinstance(cell_nodes, torch.Tensor) else cell_nodes
            nodes = [int(n) for n in raw if not (isinstance(n, float) and math.isnan(n))]
            max_d = 0.0
            for u in nodes:
                for v in nodes:
                    d = ((frac_pos[v] - frac_pos[u] - torch.round(frac_pos[v]-frac_pos[u]))
                         @ lattice.T).norm().item()
                    max_d = max(max_d, d)
            dia_vals.append(max_d)
        diameters[rank] = torch.tensor(dia_vals, dtype=frac_pos.dtype, device=device)
        print(f"[DBG-1d]   diameters[{rank}].shape = {diameters[rank].shape}")

    # -------------------------------------------------------------------------
    # 2)  Build per-adjacency feature tensors
    features: dict[str, Tensor] = {}

    for rank_pair, cell_pairs in adj.items():
        print(f"\n[DBG-2]  Processing adjacency '{rank_pair}'")
        parts = rank_pair.split("_")
        send_rank, recv_rank = parts[0], parts[1]

        # 2.1)  Raw → LongTensor indices
        s_raw, r_raw = cell_pairs
        send_idx = torch.as_tensor(s_raw, dtype=torch.long, device=device).flatten()
        recv_idx = torch.as_tensor(r_raw, dtype=torch.long, device=device).flatten()
        print(f"[DBG-2a]     raw send/rev shapes = {send_idx.shape} / {recv_idx.shape}")

        # 2.2)  Mask out-of-range
        max_send = centroids[send_rank].size(0)
        max_recv = centroids[recv_rank].size(0)
        valid = (send_idx < max_send) & (recv_idx < max_recv)
        print(f"[DBG-2b]     #valid edges = {valid.sum().item()}  "
              f"(before mask: {send_idx.numel()})")
        if valid.sum() == 0:
            print(f"[DBG-2b]     SKIP '{rank_pair}' – no valid edges")
            continue
        send_idx, recv_idx = send_idx[valid], recv_idx[valid]

        # 2.3)  Ensure 1-to-1 pairing
        if send_idx.numel() != recv_idx.numel():
            print(f"[DBG-2c]     unequal lengths – making meshgrid product")
            s, r = torch.meshgrid(send_idx, recv_idx, indexing="ij")
            send_idx, recv_idx = s.reshape(-1), r.reshape(-1)
        print(f"[DBG-2c]     paired send/rev shapes = {send_idx.shape} / {recv_idx.shape}")

        # 2.4)  Gather per-edge quantities
        c_send = centroids[send_rank][send_idx]
        c_recv = centroids[recv_rank][recv_idx]
        centroid_dist = torch.norm(c_send - c_recv, dim=1)

        d_send = diameters[send_rank][send_idx]
        d_recv = diameters[recv_rank][recv_idx]

        print(f"[DBG-2d]     centroid_dist.shape = {centroid_dist.shape}")
        print(f"[DBG-2d]     d_send.shape         = {d_send.shape}")
        print(f"[DBG-2d]     d_recv.shape         = {d_recv.shape}")

        assert centroid_dist.shape == d_send.shape == d_recv.shape, \
            f"{rank_pair}: mismatch {centroid_dist.shape} vs {d_send.shape} vs {d_recv.shape}"

        feats = torch.stack([centroid_dist, d_send, d_recv], dim=1)
        print(f"[DBG-2e]     feats.shape (3-col) = {feats.shape}")

        # 2.5)  Optional Hausdorff
        if hausdorff:
            haus_s_to_r = []
            haus_r_to_s = []
            for u_idx, v_idx in zip(send_idx.tolist(), recv_idx.tolist()):
                # grab node lists for the two cells (already int-lists, no NaNs)
                S_nodes = cell_ind[send_rank][u_idx]
                R_nodes = cell_ind[recv_rank][v_idx]

                # (1) sender → receiver
                dists_sr = []
                for u in S_nodes:
                    # find min distance to any v in R
                    dists = []
                    for v in R_nodes:
                        df = frac_pos[v] - frac_pos[u]
                        dc = (df - torch.round(df)) @ lattice.T
                        dists.append(dc.norm().item())
                    dists_sr.append(max(dists))      # sender’s worst case
                haus_s_to_r.append(max(dists_sr))     # Hausdorff S→R

                # (2) receiver → sender
                dists_rs = []
                for v in R_nodes:
                    dists = []
                    for u in S_nodes:
                        df = frac_pos[u] - frac_pos[v]
                        dc = (df - torch.round(df)) @ lattice.T
                        dists.append(dc.norm().item())
                    dists_rs.append(max(dists))      # receiver’s worst case
                haus_r_to_s.append(max(dists_rs))     # Hausdorff R→S

            h1 = torch.tensor(haus_s_to_r, dtype=frac_pos.dtype, device=device)
            h2 = torch.tensor(haus_r_to_s, dtype=frac_pos.dtype, device=device)
            feats = torch.stack([centroid_dist, d_send, d_recv, h1, h2], dim=1)
        else:
            feats = torch.stack([centroid_dist, d_send, d_recv], dim=1)

            print(f"[DBG-2f]     feats.shape (+Haus) = {feats.shape}")

        # 2.6)  Save
        features[rank_pair] = feats
        print(f"[DBG-2g]     stored features['{rank_pair}'].shape = {feats.shape}")

    # -------------------------------------------------------------------------
    print("===  END compute_invariants  ===\n")
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
        # rank_pair can be "i_j" or "i_j_via"
        parts = rank_pair.split("_")
        send_rank, recv_rank = parts[0], parts[1]

        # --------------------------------------------------------------------
        # 1) Always work with LongTensor indices
        s_raw, r_raw = cell_pairs          # two 1-D tensors *or* Python lists
        send_idx = torch.as_tensor(s_raw, dtype=torch.long,
                                   device=dev) if not torch.is_tensor(s_raw) else s_raw.to(torch.long)
        recv_idx = torch.as_tensor(r_raw, dtype=torch.long,
                                   device=dev) if not torch.is_tensor(r_raw) else r_raw.to(torch.long)

        # 2) Drop any edges that point past the end of the cell list for this rank
        max_send = centroids[send_rank].size(0)
        max_recv = centroids[recv_rank].size(0)
        valid = (send_idx < max_send) & (recv_idx < max_recv)
        if valid.sum().item() == 0:
            continue                      # nothing valid for this adjacency

        send_idx = send_idx[valid]
        recv_idx = recv_idx[valid]
        # --------------------------------------------------------------------

        # (the remainder of the loop is unchanged)
        c_send = centroids[send_rank][send_idx]
        c_recv = centroids[recv_rank][recv_idx]
        centroid_dist  = torch.norm(c_send - c_recv, dim=1)

        d_send = diameters[send_rank][send_idx]
        d_recv = diameters[recv_rank][recv_idx]

        inv_list[rank_pair] = [centroid_dist, d_send, d_recv]
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
