import torch
import torch.nn.functional as F
from torch import nn

from scale_grad import ScaleGrad
from mp_utils import AllGather
import copy

__all__ = ['simclr_loss']


class InfoNCE(nn.Module):
    """Info NCE loss (used in SimCLR)

        Parameters
        ----------
        tau : float, optional
            The value controlling the scale of cosine similarities,
            by default 0.07.
        """

    def __init__(self, tau: float = 0.07, grad_scaling=False):

        super(InfoNCE, self).__init__()
        self.tau = tau
        self.grad_scaling = grad_scaling

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, epoch) -> torch.Tensor:

        run_type = ["label_smoothing", "top_k_based_on_cos_sim", "scheduled_cos_sim", "scheduled_top_k_cos_sim"][3]

        # Collect from all gpu
        z1 = AllGather.apply(z1)
        z2 = AllGather.apply(z2)

        # Combine views and normalize
        z = torch.cat((z1, z2), dim=0)
        if self.grad_scaling:
            z = ScaleGrad.apply(z)
        z = F.normalize(z, dim=1)
        n = len(z)

        # Labels telling which images make pairs
        ones = torch.ones(n // 2).to(z.device)
        labels = ones.diagflat(n // 2) + ones.diagflat(-n // 2)

        if run_type == "label_smoothing":
            labels = labels.clamp(min=0.01)
            labels = F.normalize(labels, dim=1)

        # Note: The following code might require a large amount of memory
        # in case of large batch size
        sim_m = z @ z.T
        F.normalize(z, dim=1, p=2)
        if run_type == "top_k_based_on_cos_sim":
            # pairwise similarities between embeddings
            cos_sim_m = F.softmax(z @ z.T, dim=1)
            # masking  for ruling-out pairs from the same siamese branch
            branch_mask_zeros = torch.zeros((int(cos_sim_m.shape[0]/2), int(cos_sim_m.shape[1]/2)))
            branch_mask_ones = torch.ones((int(cos_sim_m.shape[0]/2), int(cos_sim_m.shape[1]/2)))
            branch_mask = torch.cat(
                [torch.cat([branch_mask_zeros, branch_mask_ones], dim=1),
                torch.cat([branch_mask_ones, branch_mask_zeros], dim=1)],
                dim=0
            ).to(cos_sim_m.device)
            cos_sim_m = torch.mul(branch_mask, cos_sim_m)
            # selecting top-k similarities
            top_k, top_k_indices = torch.topk(cos_sim_m, k=10, dim=1)
            top_sim = torch.zeros_like(cos_sim_m).scatter(dim=1, index=top_k_indices, src=torch.full_like(top_k, 0.75))
            labels = torch.max(labels, top_sim)
        elif run_type == "scheduled_cos_sim":
            # pairwise similarities between embeddings
            cos_sim_m = F.softmax(z @ z.T, dim=1)
            # masking  for ruling-out pairs from the same siamese branch
            branch_mask_zeros = torch.zeros((int(cos_sim_m.shape[0]/2), int(cos_sim_m.shape[1]/2)))
            branch_mask_ones = torch.ones((int(cos_sim_m.shape[0]/2), int(cos_sim_m.shape[1]/2)))
            branch_mask = torch.cat(
                [torch.cat([branch_mask_zeros, branch_mask_ones], dim=1),
                torch.cat([branch_mask_ones, branch_mask_zeros], dim=1)],
                dim=0
            ).to(cos_sim_m.device)
            cos_sim_m = torch.mul(branch_mask, cos_sim_m)
            # scheduling
            schedule_start_epoch = 30
            schedule_length_epoch = 20
            if epoch >= schedule_start_epoch:
                schedule = torch.linspace(start=0, end=1, steps=schedule_length_epoch)
                index_based_on_epoch = min(epoch - schedule_start_epoch, schedule_length_epoch - 1)
                labels = (1 - schedule[index_based_on_epoch]) * labels + schedule[index_based_on_epoch] * cos_sim_m
                labels = F.softmax(labels, dim=1)
        elif run_type == "scheduled_top_k_cos_sim":
            # hyper-parameters
            num_of_top_k = 10  # 10
            schedule_start_epoch = 30  # 30
            schedule_length_epoch = 20
            # pairwise similarities between embeddings
            cos_sim_m = F.softmax(z @ z.T, dim=1)
            # masking  for ruling-out pairs from the same siamese branch
            branch_mask_zeros = torch.zeros((int(cos_sim_m.shape[0] / 2), int(cos_sim_m.shape[1] / 2)))
            branch_mask_ones = torch.ones((int(cos_sim_m.shape[0] / 2), int(cos_sim_m.shape[1] / 2)))
            branch_mask = torch.cat(
                [torch.cat([branch_mask_zeros, branch_mask_ones], dim=1),
                 torch.cat([branch_mask_ones, branch_mask_zeros], dim=1)],
                dim=0
            ).to(cos_sim_m.device)
            cos_sim_m = torch.mul(branch_mask, cos_sim_m)
            # selecting top-k similarities
            top_k, top_k_indices = torch.topk(cos_sim_m, k=num_of_top_k, dim=1)
            top_sim = torch.zeros_like(cos_sim_m).scatter(dim=1, index=top_k_indices, src=torch.full_like(top_k, 0.75))
            labels_based_on_similarity = torch.max(labels, top_sim)
            # scheduling
            if epoch >= schedule_start_epoch:
                schedule = torch.linspace(start=0, end=1, steps=schedule_length_epoch)
                index_based_on_epoch = min(epoch - schedule_start_epoch, schedule_length_epoch - 1)
                labels = (1 - schedule[index_based_on_epoch]) * labels + schedule[index_based_on_epoch] * labels_based_on_similarity
                labels = F.normalize(labels, dim=1)

        # This is a bit of cheat. Instead of removing cells from
        # the matrix where i==j, instead we set it to a very small value
        sim_m = sim_m.fill_diagonal_(-10) / self.tau

        # Get probability distribution
        sim_true_positive = sim_m.clone()
        sim_auxiliary_positive = sim_m.clone()
        true_positive_mask = torch.roll(torch.eye(sim_m.shape[0]), int(sim_m.shape[0]/2), dims=0).bool().cuda()
        sim_auxiliary_positive.masked_fill_(true_positive_mask, (-10 / self.tau))
        sim_true_positive = torch.nn.functional.log_softmax(sim_true_positive, dim=1)
        sim_auxiliary_positive = torch.nn.functional.log_softmax(sim_auxiliary_positive, dim=1)
        sim_m = torch.where(true_positive_mask.cuda(), sim_true_positive, sim_auxiliary_positive)

        # Choose values on which we calculate the loss
        loss = -torch.sum(sim_m * labels) / n

        return loss


def simclr_loss():
    return InfoNCE()