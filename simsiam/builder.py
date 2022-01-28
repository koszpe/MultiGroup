# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

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
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
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

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class MultiGroup(SimSiam):

    def __init__(self, group_sizes, group_nums, *args, **kwargs):
        super(MultiGroup, self).__init__(*args, **kwargs)
        self.group_sizes = group_sizes
        self.group_nums = group_nums
        self.group_slices = []
        merged_gs = 0
        for gs, gn in zip(group_sizes, group_nums):
            ss_group_slices = []
            for _ in range(gn):
                ss_group_slices.append(slice(merged_gs, merged_gs + gs))
                merged_gs += gs
            self.group_slices.append(ss_group_slices)
        self.group_head = nn.Linear(self.dim, merged_gs)

    # def forward(self, x1, x2):
    #     p1, p2, z1, z2 = super(MultiGroup, self).forward(x1, x2)
    #     p1s, p2s, z1s, z2s = [], [], [], []
    #     for same_sized_groups in self.groups:
    #         ss_p1s, ss_p2s, ss_z1s, ss_z2s = [], [], [], []
    #         for group in same_sized_groups:
    #             ss_p1s.append(group(p1))
    #             ss_p2s.append(group(p2))
    #             ss_z1s.append(group(z1).detach())
    #             ss_z2s.append(group(z2).detach())
    #         p1s.append(ss_p1s)
    #         p2s.append(ss_p2s)
    #         z1s.append(ss_z1s)
    #         z2s.append(ss_z2s)
    #     return p1s, p2s, z1s, z2s


    def forward(self, x1, x2):
        p1, p2, z1, z2 = super(MultiGroup, self).forward(x1, x2)
        gs = []
        p = torch.cat([p1, p2])
        groups = self.group_head(p)
        groups = AllGather.apply(groups)
        for ss_group_slices in self.group_slices:
            ss_gs = []
            for group_slice in ss_group_slices:
                ss_gs.append(groups[:, group_slice])
            gs.append(ss_gs)
        return p1, p2, z1, z2, gs