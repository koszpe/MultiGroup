# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import device, Tensor
from torch.nn.modules.module import T

from mp_utils import AllGather


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.dim = dim
        self.pred_dim = pred_dim

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False) # output layer
                                        )
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        # p1 = z1
        # p2 = z2
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1, z2

class MultiGroup(SimSiam):

    def __init__(self, group_sizes, group_nums, *args, **kwargs):
        super(MultiGroup, self).__init__(*args, **kwargs)
        self.group_sizes = group_sizes
        self.group_nums = group_nums
        self.group_slices = []
        merged_gs = 0
        for gs, gn in zip(group_sizes, group_nums):
            self.group_slices.append(slice(merged_gs, merged_gs + gs * gn))
            merged_gs += gs * gn
        self.group_head = nn.Linear(self.dim, merged_gs)

    def forward(self, x1, x2):
        p1, p2, z1, z2 = super(MultiGroup, self).forward(x1, x2)
        gs = []
        # p = torch.cat([z1, z2])
        p = torch.cat([p1, p2])
        groups = self.group_head(p)
        groups = AllGather.apply(groups)
        bs = groups.shape[0]
        for g_slice, g_size, g_num in zip(self.group_slices, self.group_sizes, self.group_nums):
            group = groups[:, g_slice].view(bs, g_num, g_size).permute(1, 0, 2).contiguous() # group_num x batch_size x group_size
            gs.append(group)
        return p1, p2, z1.detach(), z2.detach(), gs

class MultiPredictor(nn.Module):

    def __init__(self, dim, bottleneck_dims, init_dop=0.5, min_dop=0, max_dop=0.9, dop_step=0.01):
        super().__init__()
        self.predictors = dict()
        self.init_dop = init_dop
        self.min_dop = min_dop
        self.max_dop = max_dop
        self.dop_step = dop_step
        assert len(set(bottleneck_dims)) == len(bottleneck_dims), "all bottleneck_dims must be unique"
        for bn_dim in bottleneck_dims:
            predictor = nn.Sequential(nn.Linear(dim, bn_dim, bias=False),
                                      nn.BatchNorm1d(bn_dim),
                                      nn.ReLU(inplace=True),  # hidden layer
                                      nn.Dropout(p=self.init_dop),
                                      nn.Linear(bn_dim, dim))  # output layer
            # predictor = lambda x: torch.scatter(x, -1, torch.topk(x, dim - bn_dim, dim=-1, largest=False, sorted=False)[1], 0)
            setattr(self, f"predictor_{bn_dim}", predictor)
            self.predictors[bn_dim] = predictor

    def set_dropout_p(self, p):
        self.init_dop = p
        for predictor in self.predictors.values():
            for module in predictor.modules():
                if type(module) is nn.Dropout:
                    module.p = p

    def increase_dropout_p(self, dim):
        for module in self.predictors[dim].modules():
            if type(module) is nn.Dropout:
                if module.p + self.dop_step < self.max_dop:
                    module.p += self.dop_step
                    return module.p
        return None

    def decrease_dropout_p(self, dim):
        for module in self.predictors[dim].modules():
            if type(module) is nn.Dropout:
                if module.p - self.dop_step > self.min_dop:
                    module.p -= self.dop_step
                    return module.p
        return None

    def forward(self, x):
        outs = dict()
        for bn_dim, predictor in self.predictors.items():
            outs[bn_dim] = predictor(x)
        return outs

class MultiPredictorSimSiam(SimSiam):
    def __init__(self, bottleneck_dims, init_dop, min_dop, max_dop, dop_step, *args, **kwargs):
        super(MultiPredictorSimSiam, self).__init__(*args, **kwargs)
        self.bottleneck_dims = bottleneck_dims
        self.predictor = MultiPredictor(self.dim, bottleneck_dims, init_dop, min_dop, max_dop, dop_step)

    def forward(self, x1, x2):
        p1, p2, z1, z2 = super(MultiPredictorSimSiam, self).forward(x1, x2)

        return p1, p2, z1, z2

class DoublePredHead(SimSiam):
    def __init__(self, pred_type, *args, **kwargs):
        super(DoublePredHead, self).__init__(*args, **kwargs)
        self.pred_type = pred_type
        del self.predictor
        self.predictor = lambda x: x  # Replace original predictor with identity fn
        if pred_type in ["linear", "random_linear"]:
            self.double_predictor = nn.Linear(self.dim, self.pred_dim, bias=False)
        else:
            raise NotImplementedError
        self.bn = nn.BatchNorm1d(self.pred_dim, affine=False)

    def forward(self, x1, x2):
        p1, p2, z1, z2 = super(DoublePredHead, self).forward(x1, x2)
        if self.pred_type == "random_linear":
            self.double_predictor.reset_parameters()
        all = self.double_predictor(torch.cat([p1, p2, z1, z2]))
        p1, p2, z1, z2 = torch.chunk(all, 4)
        z1 = self.bn(z1)
        z2 = self.bn(z2)
        p1 = {self.pred_dim: p1}
        p2 = {self.pred_dim: p2}
        return p1, p2, z1.detach(), z2.detach()


class DifferentPredictor(SimSiam):
    def __init__(self, pred_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_type = pred_type
        if self.pred_type != "original":
            del self.predictor
            self.predictor = nn.Sequential(nn.Linear(self.dim, self.pred_dim, bias=False),
                                                nn.Linear(self.pred_dim, self.dim, bias=True))
            if pred_type == "predefined_linear":
                root_path = "/storage/simsiam/logs/original_nobnnorelupredhead_384bs_512/"
                epoch = "0050"
                checkpoint_path = os.path.join(root_path, f"checkpoint_{epoch}.pt")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.predictor[0].weight.data.copy_(checkpoint['state_dict']['module.predictor.predictor_512.0.weight'])
                self.predictor[1].weight.data.copy_(checkpoint['state_dict']['module.predictor.predictor_512.1.weight'])
                self.predictor[1].bias.data.copy_(checkpoint['state_dict']['module.predictor.predictor_512.1.bias'])
                # self.predictor = lambda x: self.predictor(x) + torch.normal(torch.zeros_like(x), torch.ones_like(x) * 0.1)
                self.predictor.requires_grad_(False)
            elif pred_type == "low_rank_linear":
                pass
            else:
                raise NotImplementedError

    def forward(self, x1, x2):
        p1, p2, z1, z2 = super().forward(x1, x2)
        p1 = {self.pred_dim: p1}
        p2 = {self.pred_dim: p2}
        return p1, p2, z1.detach(), z2.detach()
