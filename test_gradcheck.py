#!/usr/bin/env python3
"""
Autograd gradient check for
  * etnn.invariants.compute_invariants
  * etnn.layers.ETNNLayer

Requires double precision and small input sizes to pass numerical test.
"""
import torch
from torch.autograd import gradcheck
from etnn.invariants import compute_invariants
from etnn.layers import ETNNLayer


def tiny_graph():
    frac = torch.tensor([[0.2, 0.2, 0.2],
                         [0.8, 0.2, 0.2]], dtype=torch.double, requires_grad=True)
    lattice = torch.eye(3, dtype=torch.double)
    pos = frac @ lattice.T
    cell_ind = {"0": [[0], [1]]}
    adj = {"0_0": torch.tensor([[0, 1], [1, 0]], dtype=torch.long)}
    inv = compute_invariants(pos, frac, lattice, cell_ind, adj, hausdorff=False)
    return frac, lattice, pos, cell_ind, adj, inv


def test_compute_invariants_grad():
    frac, lattice, pos, cell_ind, adj, _ = tiny_graph()

    def fn(frac_coords):
        pos_ = frac_coords @ lattice.T
        out = compute_invariants(pos_, frac_coords, lattice,
                                 cell_ind, adj, hausdorff=False)["0_0"]
        # return a scalar so gradcheck can compare Jacobian
        return out.sum()

    gradcheck(fn, (frac,), eps=1e-4, atol=1e-4, rtol=1e-2)


def test_etnnlayer_grad():
    frac, lattice, pos, cell_ind, adj, inv = tiny_graph()
    hidden = 4
    x = {"0": torch.zeros((2, hidden), dtype=torch.double, requires_grad=True)}
    layer = ETNNLayer(adjacencies=["0_0"],
                      visible_dims=[0],
                      num_hidden=hidden,
                      num_features_map={"0_0": inv["0_0"].size(1)},
                      batch_norm=False,
                      lean=True,
                      pos_update=False).double()

    def fn(feat, pos_f):
        x_in = {"0": feat}
        with torch.enable_grad():
            out_feat, _ = layer(x_in, adj, inv, pos_f @ lattice.T,
                                pos_f, lattice,
                                cell_offsets_map={"0_0": torch.zeros(2, 3, dtype=torch.long)})
        return out_feat["0"].sum()

    gradcheck(fn, (x["0"], frac), eps=1e-4, atol=1e-4, rtol=1e-2)
