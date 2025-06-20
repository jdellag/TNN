#!/usr/bin/env python3
"""
Extensivity test: clone a toy crystal N times; energy prediction
should scale ∝ N.
"""
import torch
from etnn.layers import ETNNLayer
import etnn.invariants as invariants

hidden = 8

def build_graph(rep):
    # create rep×rep×rep simple cubic super-cell of one atom per cell
    basis = torch.tensor([[0.0, 0.0, 0.0]])
    shifts = torch.stack(torch.meshgrid(
        torch.arange(rep), torch.arange(rep), torch.arange(rep),
        indexing='ij')).reshape(3, -1).T
    frac = (basis.repeat(shifts.size(0), 1) + shifts) / rep
    lattice = rep * torch.eye(3)
    pos = frac @ lattice.T
    cell_ind = {"0": [[i] for i in range(len(frac))]}
    adj = {"0_0": torch.empty(2, 0, dtype=torch.long)}

    inv = invariants.compute_invariants(pos, frac, lattice,
                                        cell_ind, adj, hausdorff=False)
    x = {"0": torch.ones((len(frac), hidden))}          # one-hot “atomic feature”
    return x, adj, inv, pos, frac, lattice

# simple “energy” head = sum of per-node scalar produced by ETNNLayer
layer = ETNNLayer(adjacencies=["0_0"], visible_dims=[0],
                  num_hidden=hidden,
                  num_features_map={"0_0": 3},
                  batch_norm=False,
                  lean=True,
                  pos_update=False)
readout = torch.nn.Linear(hidden, 1, bias=False)
torch.nn.init.ones_(readout.weight)    # so energy = sum of hidden activations

def predict_energy(rep):
    x, adj, inv, pos, frac, lat = build_graph(rep)
    with torch.no_grad():
        feat, _ = layer(x, adj, inv, pos, frac, lat,
                        cell_offsets_map={"0_0": torch.zeros(0, 3, dtype=torch.long)})
    e = readout(feat["0"]).sum()
    return e.item()

def test_extensive_scaling():
    e1 = predict_energy(1)
    e2 = predict_energy(2)
    e3 = predict_energy(3)

    assert abs(e2 - 8 * e1) < 1e-5, "2× super-cell not 8× energy!"
    assert abs(e3 - 27 * e1) < 1e-5, "3× super-cell not 27× energy!"
