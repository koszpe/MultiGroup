import torch
import torch.nn.functional as F
from torch import nn

from scale_grad import ScaleGrad
from mp_utils import AllGather

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

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        run_type = ["label_smoothing", "top_k_based_on_cos_sim"][1]

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
            cos_sim_m = F.softmax(z @ z.T, dim=1)
            branch_mask_zeros = torch.zeros((int(cos_sim_m.shape[0]/2), int(cos_sim_m.shape[1]/2)))
            branch_mask_ones = torch.ones((int(cos_sim_m.shape[0]/2), int(cos_sim_m.shape[1]/2)))
            branch_mask = torch.cat(
                [torch.cat([branch_mask_zeros, branch_mask_ones], dim=1),
                torch.cat([branch_mask_ones, branch_mask_zeros], dim=1)],
                dim=0
            ).to(cos_sim_m.device)
            cos_sim_m = torch.mul(branch_mask, cos_sim_m)

            top_k, top_k_indices = torch.topk(cos_sim_m, k=10, dim=1)
            top_sim = torch.zeros_like(cos_sim_m).scatter(dim=1, index=top_k_indices, src=torch.full_like(top_k, 0.75))
            labels = torch.max(labels, top_sim)

        # This is a bit of cheat. Instead of removing cells from
        # the matrix where i==j, instead we set it to a very small value
        sim_m = sim_m.fill_diagonal_(-10) / self.tau

        # Get probability distribution
        sim_m = torch.nn.functional.log_softmax(sim_m, dim=1)

        # Choose values on which we calculate the loss
        loss = -torch.sum(sim_m * labels) / n

        return loss


def simclr_loss():
    return InfoNCE()