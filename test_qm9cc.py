import torch
from etnn.qm9.qm9cc import QM9CC

def test_qm9cc_has_lattice_and_frac(tmp_path):
    # Instantiate with a trivial lifter (e.g. no supercell)
    dataset = QM9CC(root=str(tmp_path), 
                    lifters=["rips"], 
                    neighbor_types=["eucl"], 
                    connectivity="full",
                    supercell=False)
    data = dataset[0]   # pull the first graph

    # Check that the lattice attribute is a 3×3 tensor
    assert hasattr(data, "lattice"), "Missing data.lattice"
    assert isinstance(data.lattice, torch.Tensor)
    assert data.lattice.shape == (3, 3)

    # Check fractional coords are in [0,1)
    assert hasattr(data, "frac_pos"), "Missing data.frac_pos"
    frac = data.frac_pos
    assert torch.all(frac >= 0) and torch.all(frac < 1), \
           "Found frac_pos outside [0,1)"
