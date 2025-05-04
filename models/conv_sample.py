# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file
"""
Minimal patch for **models/conv_sample.py** to work with PyTorch‑Geometric ≥ 2.3
(including 2.6.x) while keeping every original computation unchanged.

`SparseTensor` fast‑path no longer accepts extra kwargs, so we:
1. cache `xe` on `self` inside `forward()`;
2. fetch it in `message_and_aggregate()` using `getattr`.

Public signatures and data flow remain intact — all other repo files can stay
as they were.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import PairTensor, OptTensor

from models.sampler import BEANAdjacency

# =============================================================
#  High‑level sampled convolution layer
# =============================================================
class BEANConvSample(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: Tuple[int, int, Optional[int]],
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):  # noqa: D401
        super().__init__(**kwargs)

        self.input_has_edge_channel = len(in_channels) == 3
        self.output_has_edge_channel = len(out_channels) == 3

        self.v2u_conv = BEANConvNode(
            in_channels,
            out_channels[0],
            flow="v->u",
            node_self_loop=node_self_loop,
            normalize=normalize,
            bias=bias,
            **kwargs,
        )
        self.u2v_conv = BEANConvNode(
            in_channels,
            out_channels[1],
            flow="u->v",
            node_self_loop=node_self_loop,
            normalize=normalize,
            bias=bias,
            **kwargs,
        )
        if self.output_has_edge_channel:
            self.e_conv = BEANConvEdge(
                in_channels,
                out_channels[2],
                node_self_loop=node_self_loop,
                normalize=normalize,
                bias=bias,
                **kwargs,
            )

    # ---------------------------------------------------------
    def forward(
        self,
        xu: PairTensor,
        xv: PairTensor,
        adj: BEANAdjacency,
        xe: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        xus, xut = xu
        xvs, xvt = xv
        if xe is not None:
            xe_e, xe_v2u, xe_u2v = xe
        else:
            xe_e = xe_v2u = xe_u2v = None

        out_u = self.v2u_conv((xut, xvs), adj.adj_v2u.adj, xe_v2u)
        out_v = self.u2v_conv((xus, xvt), adj.adj_u2v.adj, xe_u2v)
        out_e = None
        if self.output_has_edge_channel:
            out_e = self.e_conv((xut, xvt), adj.adj_e.adj, xe_e)
        return out_u, out_v, out_e

# =============================================================
#  Directional node convolution
# =============================================================
class BEANConvNode(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: int,
        flow: str = "v->u",
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        agg: List[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.flow = flow
        self.node_self_loop = node_self_loop
        self.normalize = normalize
        self.agg = agg or ["mean", "max"]
        self.input_has_edge_channel = len(in_channels) == 3

        n_agg = len(self.agg)
        if self.input_has_edge_channel:
            if node_self_loop:
                self.in_channels_all = (
                    in_channels[0] + n_agg * in_channels[1] + n_agg * in_channels[2]
                    if flow == "v->u"
                    else n_agg * in_channels[0] + in_channels[1] + n_agg * in_channels[2]
                )
            else:
                self.in_channels_all = (
                    n_agg * in_channels[1] + n_agg * in_channels[2]
                    if flow == "v->u"
                    else n_agg * in_channels[0] + n_agg * in_channels[2]
                )
        else:
            if node_self_loop:
                self.in_channels_all = (
                    in_channels[0] + n_agg * in_channels[1]
                    if flow == "v->u"
                    else n_agg * in_channels[0] + in_channels[1]
                )
            else:
                self.in_channels_all = n_agg * (in_channels[1] if flow == "v->u" else in_channels[0])

        self.lin = Linear(self.in_channels_all, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels) if normalize else nn.Identity()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if isinstance(self.bn, nn.BatchNorm1d):
            self.bn.reset_parameters()

    # ------------------ patched forward --------------------
    def forward(self, x: PairTensor, adj: SparseTensor, xe: OptTensor = None) -> Tensor:
        """Keep `xe=` arg, hide it from PyG."""
        assert self.input_has_edge_channel == (xe is not None)
        self._cached_xe = xe
        out = self.propagate(adj, x=x)
        out = self.lin(out)
        return self.bn(out)

    # ---------------- message & aggregate ------------------
    def message_and_aggregate(self, adj: SparseTensor, x: PairTensor) -> Tensor:  # noqa: N802
        xu, xv = x
        adj = adj.set_value(None, layout=None)
        xe: OptTensor = getattr(self, "_cached_xe", None)

        if self.flow == "v->u":
            msg_node = [matmul(adj, xv, reduce=ag) for ag in self.agg]
            if xe is not None:
                msg_edge = [scatter(xe, adj.storage.row(), dim=0, reduce=ag) for ag in self.agg]
            concat_parts = ( (xu,) if self.node_self_loop else () ) + tuple(msg_node) + (tuple(msg_edge) if xe is not None else ())
            return torch.cat(concat_parts, dim=1)

        # flow == "u->v"
        msg_node = [matmul(adj.t(), xu, reduce=ag) for ag in self.agg]
        if xe is not None:
            msg_edge = [scatter(xe, adj.storage.col(), dim=0, reduce=ag) for ag in self.agg]
        concat_parts = ( (xv,) if self.node_self_loop else () ) + tuple(msg_node) + (tuple(msg_edge) if xe is not None else ())
        return torch.cat(concat_parts, dim=1)

# =============================================================
#  Edge convolution
# =============================================================
class BEANConvEdge(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: int,
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        has_e = len(in_channels) == 3
        self.in_channels_e = in_channels[0] + in_channels[1] + (in_channels[2] if has_e else 0)
        self.lin_e = Linear(self.in_channels_e, out_channels, bias=bias)
        self.bn_e = nn.BatchNorm1d(out_channels) if normalize else nn.Identity()

    def reset_parameters(self):
        self.lin_e.reset_parameters()
        if isinstance(self.bn_e, nn.BatchNorm1d):
            self.bn_e.reset_parameters()

    # ------------------ patched forward --------------------
    def forward(self, x: PairTensor, adj: SparseTensor, xe: Tensor | None) -> Tensor:
        """edge-level convolution wrapper"""
        self._cached_xe = xe
        out_e = self.propagate(adj, x=x)
        out_e = self.lin_e(out_e)
        return self.bn_e(out_e)

    # ---------------- message & aggregate ------------------
    def message_and_aggregate(self, adj: SparseTensor, x: PairTensor) -> Tensor:  # noqa: N802
        xu, xv = x
        adj = adj.set_value(None, layout=None)
        xe: OptTensor = getattr(self, "_cached_xe", None)
        if xe is not None:
            msg_2e = torch.cat((xe, xu[adj.storage.row()], xv[adj.storage.col()]), dim=1)
        else:
            msg_2e = torch.cat((xu[adj.storage.row()], xv[adj.storage.col()]), dim=1)
        return msg_2e
