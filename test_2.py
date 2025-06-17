#!/usr/bin/env python3
import sys
import torch
from torch_geometric.nn import global_add_pool
from etnn.layers import ETNNLayer
import etnn.invariants as invariants

def build_cell_data(frac_pos, lattice):
    """
    Build cell_index, adjacency, and PBC offsets for a rank-0 graph
    by connecting any two distinct nodes whose minimum-image distance < 1.0.
    """
    N = frac_pos.size(0)
    cell_ind = {"0": [[i] for i in range(N)]}

    pairs = []
    offs  = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            df = frac_pos[j] - frac_pos[i]
            off = torch.round(df)              # integer wrap
            df_wrap = df - off
            dc = df_wrap @ lattice.T
            if dc.norm().item() < 1.0:
                pairs.append([i, j])
                offs.append(off.tolist())

    # Fallback for 2-node case if nothing found
    if len(pairs) == 0 and N == 2:
        pairs = [[0, 1], [1, 0]]
        offs  = [[0, 0, 0], [-1, 0, 0]]

    edge_index   = torch.tensor(pairs, dtype=torch.long).t()   # [2×E]
    cell_offsets = torch.tensor(offs,  dtype=torch.long)      # [E×3]
    adj = {"0_0": edge_index}
    return cell_ind, adj, cell_offsets

def main():
    # --- Unit cell (2 atoms) ---
    frac_unit   = torch.tensor([[0.1,0.0,0.0],
                                [0.9,0.0,0.0]], dtype=torch.float)
    lattice_u   = torch.eye(3)
    pos_unit    = frac_unit @ lattice_u.T

    cell_ind_u, adj_u, offs_u = build_cell_data(frac_unit, lattice_u)
    inv_u = invariants.compute_invariants(
        pos=pos_unit,
        frac_pos=frac_unit,
        lattice=lattice_u,
        cell_ind=cell_ind_u,
        adj=adj_u,
        hausdorff=False,
    )

    # --- Shared ETNNLayer setup ---
    hidden_dim = 8
    x_unit = {"0": torch.zeros((2, hidden_dim))}
    layer = ETNNLayer(
        adjacencies=["0_0"],
        visible_dims=[0],
        num_hidden=hidden_dim,
        num_features_map={"0_0": inv_u["0_0"].shape[1]},
        batch_norm=False,
        lean=True,
        pos_update=False,
    )
    layer.eval()

    # Zero all Linear weights/biases for determinism
    for m in layer.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Forward on unit cell
    with torch.no_grad():
        x1, pos1 = layer(
            x_unit, adj_u, inv_u, pos_unit,
            frac_unit, lattice_u, {"0_0": offs_u},
        )
        out1 = global_add_pool(x1["0"], torch.zeros(2, dtype=torch.long))

    # --- Supercell (2×2 replication) ---
    # Offsets: (0,0), (1,0), (0,1), (1,1) in x-y plane
    tiles = [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
    frac_sup_list = []
    for dx,dy,dz in tiles:
        shift = torch.tensor([dx,dy,dz], dtype=torch.float)
        for f in frac_unit:
            frac_sup_list.append(((f + shift) % 1.0).tolist())
    frac_sup  = torch.tensor(frac_sup_list, dtype=torch.float)   # [8×3]
    lattice_s = 2.0 * torch.eye(3)
    pos_sup   = frac_sup @ lattice_s.T

    cell_ind_s, adj_s, offs_s = build_cell_data(frac_sup, lattice_s)
    inv_s = invariants.compute_invariants(
        pos=pos_sup,
        frac_pos=frac_sup,
        lattice=lattice_s,
        cell_ind=cell_ind_s,
        adj=adj_s,
        hausdorff=False,
    )

    x_sup = {"0": torch.zeros((frac_sup.size(0), hidden_dim))}
    with torch.no_grad():
        x2, pos2 = layer(
            x_sup, adj_s, inv_s, pos_sup,
            frac_sup, lattice_s, {"0_0": offs_s},
        )
        out2 = global_add_pool(x2["0"], torch.zeros(frac_sup.size(0), dtype=torch.long))

    feats_ok = torch.allclose(out1, out2, atol=1e-6)

    if feats_ok:
        print("PASS: supercell vs unit-cell consistency under PBC")
        sys.exit(0)
    else:
        print("FAIL: pooled outputs differ!")
        print(" unit pooled:", out1)
        print(" super pooled:", out2)
        sys.exit(1)

if __name__ == "__main__":
    main()
