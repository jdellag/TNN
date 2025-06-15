#!/usr/bin/env python3
import sys
import torch
from etnn.layers import ETNNLayer

def main():
    # 1) Build a two‐atom “crystal” in fractional coords
    frac_pos = torch.tensor([[0.1, 0.0, 0.0],
                             [0.9, 0.0, 0.0]], dtype=torch.float)
    lattice = torch.eye(3)
    pos = frac_pos @ lattice.T

    # 2) Dummy features
    num_hidden = 4
    num_inv    = 1
    x = torch.zeros((2, num_hidden))

    # 3) Single adjacency “0_0”
    adj = {"0_0": torch.tensor([[0, 1], [1, 0]], dtype=torch.long)}

    # 4) Dummy invariant
    inv = {"0_0": torch.tensor([[0.2], [0.2]], dtype=torch.float)}

    # 5) Cell‐offsets map
    cell_offsets = {"0_0": torch.tensor([[ 0,0,0], [-1,0,0]], dtype=torch.long)}

    # 6) Instantiate layer
    layer = ETNNLayer(
        adjacencies=["0_0"],
        visible_dims=[0],
        num_hidden=num_hidden,
        num_features_map={"0_0": num_inv},
        batch_norm=False,
        lean=True,
        pos_update=True,
    )
    layer.eval()

    # First forward
    with torch.no_grad():
        x1, pos1 = layer(
            x={"0": x},
            adj=adj,
            inv=inv,
            pos=pos,
            frac_pos=frac_pos,
            lattice=lattice,
            cell_offsets_map=cell_offsets,
        )

    # Translate by one cell
    frac_pos2 = (frac_pos + torch.tensor([1.0, 0.0, 0.0])) % 1.0
    pos2      = frac_pos2 @ lattice.T
    cell_offsets2 = {"0_0": torch.tensor([[0,0,0],[0,0,0]], dtype=torch.long)}

    with torch.no_grad():
        x2, pos2_out = layer(
            x={"0": x},
            adj=adj,
            inv=inv,
            pos=pos2,
            frac_pos=frac_pos2,
            lattice=lattice,
            cell_offsets_map=cell_offsets2,
        )

    # Check invariance of features, and allow small float‐rounding in pos
    feats_ok = torch.allclose(x1["0"], x2["0"], atol=1e-6)
    pos_ok   = torch.allclose(pos1, pos2_out, atol=1e-3)

    if feats_ok and pos_ok:
        print("PASS: layer is translation‐equivariant under PBC")
        sys.exit(0)
    else:
        if not feats_ok:
            print("FAIL: feature mismatch")
            print(" x1:", x1["0"])
            print(" x2:", x2["0"])
        if not pos_ok:
            print("FAIL: position mismatch (within tolerance)")
            print(" pos1:", pos1)
            print(" pos2:", pos2_out)
        sys.exit(1)

if __name__ == "__main__":
    main()
