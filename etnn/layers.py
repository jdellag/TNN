from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from etnn import utils


class ETNNLayer(nn.Module):
    def __init__(
        self,
        adjacencies: List[str],
        visible_dims: List[int],
        num_hidden: int,
        num_features_map: Dict[str, int],
        batch_norm: bool = False,
        lean: bool = True,
        pos_update: bool = False,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.visible_dims = visible_dims
        self.num_features_map = num_features_map
        self.batch_norm = batch_norm
        self.lean = lean
        self.pos_update = pos_update

        # message‐passing modules for each adjacency type
        self.message_passing = nn.ModuleDict({
            adj: BaseMessagePassingLayer(
                num_hidden,
                self.num_features_map[adj],
                batch_norm=batch_norm,
                lean=lean,
            )
            for adj in adjacencies
        })

        # state update MLPs by cell‐rank
        self.update = nn.ModuleDict()
        for dim in visible_dims:
            factor = 1 + sum(adj_type[2] == str(dim) for adj_type in adjacencies)
            layers = [nn.Linear(factor * num_hidden, num_hidden)]
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(num_hidden))
            if not self.lean:
                extra = [nn.SiLU(), nn.Linear(num_hidden, num_hidden)]
                if self.batch_norm:
                    extra.append(nn.BatchNorm1d(num_hidden))
                layers.extend(extra)
            self.update[str(dim)] = nn.Sequential(*layers)

        # optional position‐update weights
        if pos_update:
            self.pos_update_wts = nn.Linear(num_hidden, 1, bias=False)
            nn.init.trunc_normal_(self.pos_update_wts.weight, std=0.02)

    def radial_pos_update(
        self,
        pos: Tensor,
        mes: Dict[str, Tensor],
        adj: Dict[str, Tensor],
        frac_pos: Tensor,
        lattice: Tensor,
        cell_offsets_map: Dict[str, Tensor],
    ) -> Tensor:
        """
        PBC‐aware coordinate update for 0→0 adjacency only.
        """
        # locate the node‐to‐node adjacency key
        key = next(k for k in adj if k[0] == "0" and k[2] == "0")
        send, recv = adj[key]                          # each [E]
        offsets = cell_offsets_map[key]                # [E×3]

        # edge weights from messages at receiver
        wts = self.pos_update_wts(mes[key][recv])      # [E×1]

        # minimum‐image Δf in frac‐space, then apply integer offsets
        df = frac_pos[send] - frac_pos[recv]           # [E×3]
        df_wrap = df - torch.round(df)                 # wrap to (–0.5,0.5]
        df_wrap = df_wrap + offsets.to(df_wrap)        # apply cell shifts

        # back to Cartesian
        dc = df_wrap @ lattice.T                       # [E×3]

        # scatter‐add weighted displacements
        delta = utils.scatter_add(
            dc * wts,                                  # [E×3] × [E×1] → [E×3]
            send,
            dim=0,
            dim_size=pos.size(0),
        )
        return pos + 0.1 * delta

    def forward(
        self,
        x: Dict[str, Tensor],
        adj: Dict[str, Tensor],
        inv: Dict[str, Tensor],
        pos: Tensor,
        frac_pos: Tensor,
        lattice: Tensor,
        cell_offsets_map: Dict[str, Tensor],
    ) -> (Dict[str, Tensor], Tensor):
        # 1) Compute messages for each adjacency
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]),
                index=adj[adj_type],
                edge_attr=inv[adj_type],
            )
            for adj_type in self.adjacencies
        }

        # 2) State update per rank via concatenation + MLP + residual
        h = {
            dim: torch.cat(
                [x[dim]] +
                [mes[a] for a in self.adjacencies if a[2] == dim],
                dim=1,
            )
            for dim in x
        }
        h = {dim: self.update[dim](h_dim) for dim, h_dim in h.items()}
        x = {dim: x[dim] + h[dim] for dim in x}

        # 3) Optional position update (PBC‐aware)
        if self.pos_update:
            pos = self.radial_pos_update(
                pos, mes, adj, frac_pos, lattice, cell_offsets_map
            )

        return x, pos


# Keep your BaseMessagePassingLayer below unchanged
class BaseMessagePassingLayer(nn.Module):
    def __init__(self, num_hidden, num_inv, batch_norm: bool = False, lean: bool = True):
        super().__init__()
        self.batch_norm = batch_norm
        self.lean = lean
        layers = [
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
        ]
        if self.batch_norm:
            layers.insert(1, nn.BatchNorm1d(num_hidden))
        if not self.lean:
            extra = [
                nn.Linear(num_hidden, num_hidden),
                nn.SiLU(),
            ]
            if self.batch_norm:
                extra.insert(1, nn.BatchNorm1d(num_hidden))
            layers.extend(extra)
        self.message_mlp = nn.Sequential(*layers)
        self.edge_inf_mlp = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())

    def forward(self, x, index, edge_attr):
        send, recv = index
        x_send, x_recv = x
        m_send, m_recv = x_send[send], x_recv[recv]
        state = torch.cat((m_send, m_recv, edge_attr), dim=1)
        msgs = self.message_mlp(state)
        wts = self.edge_inf_mlp(msgs)
        return utils.scatter_add(msgs * wts, recv, dim=0, dim_size=x_recv.size(0))

class BaseMessagePassingLayer(nn.Module):
    def __init__(
        self, num_hidden, num_inv, batch_norm: bool = False, lean: bool = True
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.lean = lean
        message_mlp_layers = [
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
        ]
        if self.batch_norm:
            message_mlp_layers.insert(1, nn.BatchNorm1d(num_hidden))

        if not self.lean:
            extra_layers = [
                nn.Linear(num_hidden, num_hidden),
                nn.SiLU(),
            ]
            if self.batch_norm:
                extra_layers.insert(1, nn.BatchNorm1d(num_hidden))
            message_mlp_layers.extend(extra_layers)
        self.message_mlp = nn.Sequential(*message_mlp_layers)
        self.edge_inf_mlp = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = utils.scatter_add(
            messages * edge_weights, index_rec, dim=0, dim_size=x_rec.shape[0]
        )

        return messages_aggr
