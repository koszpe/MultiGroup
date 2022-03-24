import torch
import torch.autograd


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # norm = torch.norm(x, p=2, dim=1, keepdim=True)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # return grad_output * norm ** 2 / norm.mean() ** 2
        # return grad_output * torch.sqrt(x ** 3) / torch.sqrt(x.mean() ** 3)
        # x = (x ** 2).sum(dim=-1, keepdim=True)
        # return grad_output * torch.sqrt(x ** 3) / torch.sqrt(x.mean() ** 3)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # grad_mean = grad_output.mean()
        # scaled_grad = grad_output * norm ** 2
        # scaled_grad_mean = scaled_grad.mean()
        # rescaled_grad = scaled_grad * (grad_mean / scaled_grad_mean)
        #
        # # print(f"orig: {grad_mean.item()} scaled: {scaled_grad_mean.item()} ratio: {(grad_mean / scaled_grad_mean)} rescaled: {rescaled_grad.mean().item()}")
        # return rescaled_grad
        # return grad_output * norm ** 2 / norm.mean() ** 2
        return grad_output * norm ** 2 / 20 ** 2
