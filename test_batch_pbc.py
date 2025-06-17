#!/usr/bin/env python3
import sys
import torch
from torch_geometric.loader import DataLoader
from etnn.combinatorial_data import CombinatorialComplexData

def main():
    # Graph 1: 2 nodes
    frac1 = torch.tensor([[0.1, 0.0, 0.0],
                          [0.9, 0.0, 0.0]], dtype=torch.float)
    lat1  = torch.eye(3)
    offs1 = torch.tensor([[ 0, 0, 0],
                          [-1, 0, 0]], dtype=torch.long)

    # Graph 2: 3 nodes
    frac2 = torch.tensor([[0.2, 0.0, 0.0],
                          [0.8, 0.0, 0.0],
                          [0.5, 0.0, 0.0]], dtype=torch.float)
    lat2  = 2.0 * torch.eye(3)
    offs2 = torch.tensor([[ 0, 0, 0],
                          [-1, 0, 0],
                          [ 0, 0, 0]], dtype=torch.long)

    # Build CombinatorialComplexData instances
    data1 = CombinatorialComplexData(
        frac_pos=frac1,
        lattice=lat1,
        cell_offsets_0_1=offs1
    )
    data2 = CombinatorialComplexData(
        frac_pos=frac2,
        lattice=lat2,
        cell_offsets_0_1=offs2
    )

    # Batch them
    loader = DataLoader([data1, data2], batch_size=2)
    batch = next(iter(loader))

    # Assertions
    assert batch.lattice.shape == (2, 3, 3), f"lattice shape {batch.lattice.shape}"
    expected_frac_len = frac1.size(0) + frac2.size(0)
    assert batch.frac_pos.size(0) == expected_frac_len, (
        f"frac_pos length {batch.frac_pos.size(0)}, expected {expected_frac_len}"
    )
    expected_offs_len = offs1.size(0) + offs2.size(0)
    assert batch.cell_offsets_0_1.size(0) == expected_offs_len, (
        f"cell_offsets_0_1 length {batch.cell_offsets_0_1.size(0)}, expected {expected_offs_len}"
    )

    print("PASS: batching of frac_pos, lattice, and cell_offsets works correctly")
    sys.exit(0)

if __name__ == "__main__":
    main()
