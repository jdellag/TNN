#!/usr/bin/env python3
import sys
import torch
from torch_geometric.data import Data
from etnn.qm9.lifts.rips_vietoris_complex import rips_lift

def test_edge_pbc():
    # 2 points at frac x = 0.1 and 0.9 → should be neighbors under PBC
    frac_pos = torch.tensor([[0.1, 0.0, 0.0],
                             [0.9, 0.0, 0.0]], dtype=torch.float)
    lattice  = torch.eye(3)
    x = torch.zeros((2, 1))
    graph = Data(x=x, frac_pos=frac_pos, lattice=lattice)

    simplexes = rips_lift(graph, dim=1, dis=0.3, fc_nodes=False)
    edges = {frozenset(s) for (s, _) in simplexes if len(s) == 2}
    return frozenset({0, 1}) in edges

def test_triangle_pbc():
    # 3 points at frac x = 0.0, 0.9, 0.1 → all three pairwise distances ≤ 0.3 under PBC
    frac_pos = torch.tensor([[0.0, 0.0, 0.0],
                             [0.9, 0.0, 0.0],
                             [0.1, 0.0, 0.0]], dtype=torch.float)
    lattice  = torch.eye(3)
    x = torch.zeros((3, 1))
    graph = Data(x=x, frac_pos=frac_pos, lattice=lattice)

    simplexes = rips_lift(graph, dim=2, dis=0.3, fc_nodes=False)
    triangles = {frozenset(s) for (s, _) in simplexes if len(s) == 3}
    return frozenset({0, 1, 2}) in triangles

def main():
    ok1 = test_edge_pbc()
    ok2 = test_triangle_pbc()

    if ok1 and ok2:
        print("PASS: PBC 1‐simplices and 2‐simplices generated correctly")
        sys.exit(0)
    else:
        if not ok1:
            print("FAIL: edge under PBC missing")
        if not ok2:
            print("FAIL: triangle under PBC missing")
        sys.exit(1)

if __name__ == "__main__":
    main()
