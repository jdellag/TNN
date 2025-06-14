#!/usr/bin/env python3
import sys
import torch
from etnn.invariants import compute_invariants

def main():
    # 2 nodes at frac x=0.1 and 0.9, unit cell. Cutoff irrelevant for diameters.
    frac_pos = torch.tensor([[0.1, 0.0, 0.0],
                             [0.9, 0.0, 0.0]], dtype=torch.float)
    lattice = torch.eye(3)
    pos = frac_pos @ lattice.T

    # Define cells: one 1-cell (edge) containing nodes [0,1]
    cell_ind = {"1": [[0, 1]]}
    # Adjacency: compare the 1-cell to itself
    adj = {"1_1": torch.tensor([[0], [0]], dtype=torch.long)}

    inv = compute_invariants(
        pos=pos,
        frac_pos=frac_pos,     # if you updated the signature
        lattice=lattice,       # likewise
        cell_ind=cell_ind,
        adj=adj,
        hausdorff=False,
    )

    # Extract the features for the one adjacency pair
    features = inv["1_1"][0]   # shape [num_features], here num_features=3
    # features = [centroid_dist, diameter_send, diameter_recv]

    diameter_send = features[1].item()
    diameter_recv = features[2].item()

    ok = True
    if abs(diameter_send - 0.2) > 1e-6:
        print(f"FAIL: diameter_send = {diameter_send:.6f}, expected 0.2")
        ok = False
    if abs(diameter_recv - 0.2) > 1e-6:
        print(f"FAIL: diameter_recv = {diameter_recv:.6f}, expected 0.2")
        ok = False

    if ok:
        print("PASS: diameters under PBC computed correctly")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
