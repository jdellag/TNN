import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from etnn.layers import ETNNLayer
from etnn import utils, invariants


class ETNN(nn.Module):
    """
    The E(n)-Equivariant Topological Neural Network (ETNN) model.
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        hausdorff_dists: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.0,
        lean: bool = True,
        global_pool: bool = False,  # whether or not to use global pooling
        sparse_invariant_computation: bool = False,
        sparse_agg_max_cells: int = 100,  # maximum size to consider for diameter and hausdorff dists
        pos_update: bool = False,  # performs the equivariant position update, optional
    ) -> None:
        super().__init__()

        self.initial_features = initial_features

        # make inv_fts_map for backward compatibility
        self.num_invariants = 5 if hausdorff_dists else 3
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.batch_norm = batch_norm
        self.lean = lean
        max_dim = max(num_features_per_rank.keys())
        self.global_pool = global_pool
        self.visible_dims = list(range(max_dim + 1))
        self.pos_update = pos_update
        self.dropout = dropout

        # params for invariant computation
        self.sparse_invariant_computation = sparse_invariant_computation
        self.sparse_agg_max_cells = sparse_agg_max_cells
        self.hausdorff = hausdorff_dists
        self.cell_list_fmt = "list"# if sparse_invariant_computation else "padded"

        if sparse_invariant_computation:
            self.inv_fun = invariants.compute_invariants_sparse
        else:
            self.inv_fun = invariants.compute_invariants

        # keep only adjacencies that are compatible with visible_dims
        if visible_dims is not None:
            self.adjacencies = []
            for adj in adjacencies:
                max_rank = max(int(rank) for rank in adj.split("_")[:2])
                if max_rank in visible_dims:
                    self.adjacencies.append(adj)
        else:
            self.visible_dims = list(range(max_dim + 1))
            self.adjacencies = adjacencies

        # layers
        if self.normalize_invariants:
            self.inv_normalizer = nn.ModuleDict(
                {
                    adj: nn.BatchNorm1d(self.num_inv_fts_map[adj], affine=False)
                    for adj in self.adjacencies
                }
            )

        embedders = {}
        for dim in self.visible_dims:
            embedder_layers = [nn.Linear(num_features_per_rank[dim], num_hidden)]
            if self.batch_norm:
                embedder_layers.append(nn.BatchNorm1d(num_hidden))
            embedders[str(dim)] = nn.Sequential(*embedder_layers)
        self.feature_embedding = nn.ModuleDict(embedders)

        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                    self.batch_norm,
                    self.lean,
                    self.pos_update,
                )
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()

        for dim in self.visible_dims:
            if self.global_pool:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_hidden),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_hidden)
            else:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_out),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_out)

        if self.global_pool:
            self.post_pool = nn.Sequential(
                nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_out),
            )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device

        # 1) Build cell indices for geometric features
        cell_ind = {
            str(i): graph.cell_list(i, format="list")
            for i in self.visible_dims
        }

        # 2) Load adjacency index lists
        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"adj_{adj_type}")
        }

        # 3) Compute initial features (centroids, memberships, hetero)
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    features[feature_type][str(i)] = invariants.compute_centroids(
                        cell_ind[str(i)], graph.x
                    )
                elif feature_type == "mem":
                    mem = {i: getattr(graph, f"mem_{i}") for i in self.visible_dims}
                    features[feature_type][str(i)] = mem[i].float()
                elif feature_type == "hetero":
                    features[feature_type][str(i)] = getattr(graph, f"x_{i}")

        x = {
            str(i): torch.cat(
                [features[ft][str(i)] for ft in self.initial_features],
                dim=1,
            )
            for i in self.visible_dims
        }

        # 4) Prepare kwargs for invariant computation
        inv_comp_kwargs = {
            "cell_ind": cell_ind,
            "adj": adj,
            "hausdorff": self.hausdorff,
        }
        if self.sparse_invariant_computation:
            agg_indices, _ = invariants.sparse_computation_indices_from_cc(
                cell_ind, adj, self.sparse_agg_max_cells
            )
            inv_comp_kwargs["rank_agg_indices"] = agg_indices
        for adj_type, idx in adj.items():
            assert idx.dtype == torch.long, f"{adj_type} is {idx.dtype}"
        # 5) Embed features and compute invariants
        pos = graph.pos
        x = {dim: self.feature_embedding[dim](feat) for dim, feat in x.items()}
        # Compute E(n) invariants, now including PBC info
        inv = self.inv_fun(
            pos,
            frac_pos=graph.frac_pos,
            lattice=graph.lattice,
            **inv_comp_kwargs,
        )
        if self.normalize_invariants:
            inv = {
                at: self.inv_normalizer[at](feat) for at, feat in inv.items()
            }

        # ── NEW: Build the PBC cell‐offsets map for message‐passing ──
        # make sure BOTH edge_index_* and cell_offsets_* exist
        # if they are missing, we fabricate empty/zero tensors so that the
        # rest of the pipeline can still run (no rank-0 edges for that graph).
        # ------------------------------------------------------------------
        cell_offsets_map = {}
        for at in self.adjacencies:
            ei_key  = f"edge_index_{at}"
            off_key = f"cell_offsets_{at}"
        
        #    if not hasattr(graph, ei_key):
        #         # No edges of this type in this batch → create a dummy [2,0] tensor
        #        empty_ei = torch.empty(2, 0,
        #                                dtype=torch.long,
        #                                device=graph.x.device)
        #        setattr(graph, ei_key, empty_ei)
            if not hasattr(graph, ei_key):
                # No edges of this type in this batch → create a dummy [2,0] tensor
                dev = _infer_device(graph)
                empty_ei = torch.empty(2, 0, dtype=torch.long, device=dev)
                setattr(graph, ei_key, empty_ei)
            edge_index = getattr(graph, ei_key)                      # [2, E]
            E = edge_index.size(1)
        
            if hasattr(graph, off_key):
                cell_offsets_map[at] = getattr(graph, off_key)       # [E, 3]
            else:
                # fabricate zero-translations for all E edges
                cell_offsets_map[at] = edge_index.new_zeros(E, 3)
                setattr(graph, off_key, cell_offsets_map[at])        # cache
        # ────────────────────────────────────────────────────────────

        # 6) Message‐passing layers (with PBC‐aware geometry)
        for layer in self.layers:
            x, pos = layer(
                x,
                adj,
                inv,
                pos,
                graph.frac_pos,
                graph.lattice,
                cell_offsets_map,
            )
            # If positions are updated, recompute invariants
            if self.pos_update:
                inv = self.inv_fun(pos, **inv_comp_kwargs)
                if self.normalize_invariants:
                    inv = {
                        at: self.inv_normalizer[at](feat)
                        for at, feat in inv.items()
                    }
            # Optional dropout
            if self.dropout > 0:
                x = {
                    dim: nn.functional.dropout(feat, p=self.dropout)
                    for dim, feat in x.items()
                }

        # 7) Readout / pooling
        out = {dim: self.pre_pool[dim](feat) for dim, feat in x.items()}
        if self.global_pool:
            cell_batch = {
                str(i): utils.slices_to_pointer(graph._slice_dict[f"slices_{i}"])
                for i in self.visible_dims
            }
            out = {
                dim: global_add_pool(out[dim], cell_batch[dim])
                for dim in out
            }
            state = torch.cat(tuple(out.values()), dim=1)
            out = torch.squeeze(self.post_pool(state), -1)

        return out

    def __str__(self):
        return f"ETNN ({self.type})"
def _infer_device(g):
    """Return a torch.device that definitely exists in `g` (never None)."""
    if getattr(g, 'pos', None) is not None:        # Geo datasets always have it
        return g.pos.device
    if getattr(g, 'batch', None) is not None:
        return g.batch.device
    # last resort – search for any tensor
    for v in g.__dict__.values():
        if torch.is_tensor(v):
            return v.device
    return torch.device('cpu')