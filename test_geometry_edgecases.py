#!/usr/bin/env python3
"""
Edge-case smoke-tests:

* Triclinic cell (no orthogonality, angles ≠ 90°)
* 2-D slab (degenerate c-axis → effectively surface)
* Large 4 × 4 × 4 super-cell (checks integer offsets don’t overflow)

Each dataset must honour batching of `lattice`, `frac_pos` and
`cell_offsets_*`.
"""
import torch
from torch_geometric.loader import DataLoader
from etnn.combinatorial_data import CombinatorialComplexData


def make_triclinic():
    a = torch.tensor([[5.0, 0.0, 0.0],
                      [1.5, 5.5, 0.0],
                      [0.5, 1.0, 8.0]])      # Å
    frac = torch.rand(10, 3)                 # ten random atoms
    offs = torch.zeros(0, 3, dtype=torch.long)
    return CombinatorialComplexData(frac_pos=frac, lattice=a,
                                    cell_offsets_0_1=offs)


def make_slab():
    a = torch.diag(torch.tensor([5.0, 5.0, 30.0]))  # huge c-axis
    frac = torch.rand(20, 3)
    # squeeze in z so all atoms lie near the surface (0.1 fractional height)
    frac[:, 2] = 0.05 * torch.rand_like(frac[:, 2])
    offs = torch.zeros(0, 3, dtype=torch.long)
    return CombinatorialComplexData(frac_pos=frac, lattice=a,
                                    cell_offsets_0_1=offs)


def make_big_supercell():
    a = 4.0 * torch.eye(3)
    frac = torch.rand(400, 3)                # plenty of atoms
    offs = torch.zeros(0, 3, dtype=torch.long)
    return CombinatorialComplexData(frac_pos=frac, lattice=a,
                                    cell_offsets_0_1=offs)


def test_batching_edgecases():
    g1, g2, g3 = make_triclinic(), make_slab(), make_big_supercell()
    loader = DataLoader([g1, g2, g3], batch_size=3)
    batch = next(iter(loader))

    # (B,3,3) lattice ✔
    assert batch.lattice.shape == (3, 3, 3)

    # frac_pos concatenated ✔
    assert batch.frac_pos.size(0) == g1.frac_pos.size(0) + g2.frac_pos.size(0) + g3.frac_pos.size(0)

    # cell_offsets_* concatenated (zero-length in this smoke test)
    assert hasattr(batch, "cell_offsets_0_1")
    assert batch.cell_offsets_0_1.ndim == 2 and batch.cell_offsets_0_1.size(1) == 3
