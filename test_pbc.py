# test_pbc.py

import torch
from etnn.qm9.qm9cc import QM9CC

# 1) SMOKE TEST: does QM9CC(supercell=True) produce cell_offsets?
if __name__ == "__main__":
    # 3×3 identity cell (Å); adjust size as desired
    lattice = torch.eye(3) * 10.0

    ds = QM9CC(
        root="path/to/qm9",             # adjust to where you keep QM9
        lifters=["rips"],              # or whatever lifter(s) you use
        neighbor_types=["rips"],        # match your lifters
        connectivity="self",        # match your config
        supercell=True,
        lifter_kwargs={
            "lattice": lattice,
            "cutoff": 5.0,              # in Å
        },
    )

    data = ds[0]
    print("edge_index shape:", data.edge_index.shape)
    print("cell_offsets shape:", getattr(data, "cell_offsets", None).shape)
    # print first few offsets
    print("sample offsets:\n", data.cell_offsets[:10])


    # 2) BOUNDARY TEST: two points at f = [0.99,0.5,0.5] and [0.01,0.5,0.5]
    from torch_geometric.data import Data
    from etnn.lifter import Lifter, get_adjacency_types
    from etnn.lifter import CombinatorialComplexTransform
    from etnn.qm9.lifts.registry import LIFTER_REGISTRY

    dim = 3
    # build a Lifter that uses the same 'rips' lifter + supercell
    lifter = Lifter(
        lifters=["rips", f"supercell:{dim+1}"],
        registry=LIFTER_REGISTRY,
        dim=dim,
        lattice=lattice,
        cutoff=1.5,
    )
    adj = get_adjacency_types(dim+1, "self", ["max"])
    transform = CombinatorialComplexTransform(lifter=lifter, adjacencies=adj)

    # fractional coords near opposite faces
    frac = torch.tensor([[0.99, 0.5, 0.5], [0.01, 0.5, 0.5]])
    cart = frac @ lattice.T
    toy = Data(pos=cart)

    toy = transform(toy)
    print("\nBOUNDARY TEST")
    print("  edge_index:\n", toy.edge_index)
    print("  cell_offsets:\n", toy.cell_offsets)
    # you should see an edge between nodes 0↔1 and an offset of ±[1,0,0]
