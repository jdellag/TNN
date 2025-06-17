#!/usr/bin/env python3
import sys
import torch
from torch_geometric.nn import global_add_pool
from etnn.layers import ETNNLayer
import etnn.invariants as invariants

def main():
    # 1) Unit‐cell: two atoms just across the boundary
    frac_pos = torch.tensor([[0.1, 0.0, 0.0],
                             [0.9, 0.0, 0.0]], dtype=torch.float)
    lattice  = torch.eye(3)
    pos       = frac_pos @ lattice.T

    # 2) Cell indices for rank‐0 (each atom is its own “cell”)
    cell_ind = {"0": [[0], [1]]}

    # 3) Adjacency “0_0”: bidirectional edges 0→1 and 1→0
    adj = {
        "0_0": torch.tensor([[0, 1],
                             [1, 0]], dtype=torch.long)
    }

    # 4) Compute PBC‐aware invariants (skip Hausdorff here)
    inv = invariants.compute_invariants(
        pos=pos,
        frac_pos=frac_pos,
        lattice=lattice,
        cell_ind=cell_ind,
        adj=adj,
        hausdorff=False,
    )

    # 5) Dummy initial features: one scalar per node
    #    (ETNNLayer will expect hidden‐dim inputs, so we embed to zeros)
    hidden_dim = 8
    x = {"0": torch.zeros((2, hidden_dim))}

    # 6) Build one ETNNLayer (no batch‐norm, lean, no pos_update)
    layer = ETNNLayer(
        adjacencies=["0_0"],
        visible_dims=[0],
        num_hidden=hidden_dim,
        num_features_map={"0_0": inv["0_0"].shape[1]},
        batch_norm=False,
        lean=True,
        pos_update=False,
    )
    layer.eval()

    # 7) Build the cell‐offsets map for PBC
    cell_offsets = {
        # For edge 0→1 we need no wrap; for 1→0 we wrap by -1 in x
        "0_0": torch.tensor([[ 0, 0, 0],
                             [-1, 0, 0]], dtype=torch.long)
    }

    # 8) First forward pass
    with torch.no_grad():
        x1, pos1 = layer(
            x,
            adj,
            inv,
            pos,
            frac_pos,
            lattice,
            cell_offsets,
        )
        out1 = global_add_pool(x1["0"], torch.zeros(2, dtype=torch.long))

    # 9) Translate fractional coords by +1 in x (wraps back)
    frac_pos2 = (frac_pos + torch.tensor([1.0, 0.0, 0.0])) % 1.0
    pos2      = frac_pos2 @ lattice.T
    # Offsets drop the -1 wrap once we translate
    cell_offsets2 = {
        "0_0": torch.tensor([[0, 0, 0],
                             [0, 0, 0]], dtype=torch.long)
    }
    inv2 = invariants.compute_invariants(
        pos=pos2,
        frac_pos=frac_pos2,
        lattice=lattice,
        cell_ind=cell_ind,
        adj=adj,
        hausdorff=False,
    )
    with torch.no_grad():
        x2, pos2_out = layer(
            x,
            adj,
            inv2,
            pos2,
            frac_pos2,
            lattice,
            cell_offsets2,
        )
        out2 = global_add_pool(x2["0"], torch.zeros(2, dtype=torch.long))

    # 10) Check invariance (features & pooled graph vector)
    feats_ok = torch.allclose(out1, out2, atol=1e-6)
    pos_ok   = torch.allclose(pos1, pos2_out, atol=1e-6)

    if feats_ok and pos_ok:
        print("PASS: end-to-end translation‐equivariance under PBC")
        sys.exit(0)
    else:
        if not feats_ok:
            print("FAIL: model output drifted:")
            print(" out1:", out1)
            print(" out2:", out2)
        if not pos_ok:
            print("FAIL: position drifted:")
            print(" pos1:", pos1)
            print(" pos2:", pos2_out)
        sys.exit(1)

if __name__ == "__main__":
    main()
