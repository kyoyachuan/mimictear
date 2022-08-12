import torch
from torch.nn import functional as F

from .contants import LossType


class MinimaxLoss:
    def g_loss(self, fake_logits):
        target = torch.ones_like(fake_logits).cuda()
        return F.binary_cross_entropy_with_logits(fake_logits, target)

    def d_loss(self, real_logits, fake_logits):
        target_ones = torch.ones_like(real_logits).cuda()
        target_zeros = torch.zeros_like(fake_logits).cuda()
        loss = F.binary_cross_entropy_with_logits(real_logits, target_ones)
        loss += F.binary_cross_entropy_with_logits(fake_logits, target_zeros)
        return loss


class WassersteinLoss:
    def g_loss(self, fake_logits):
        return -fake_logits.mean()

    def d_loss(self, real_logits, fake_logits):
        return -(real_logits.mean() - fake_logits.mean())

    def clamp_params(self, params, clamp_values):
        for param in params:
            param.data.clamp_(-clamp_values, clamp_values)


class HingeLoss(WassersteinLoss):
    def d_loss(self, real_logits, fake_logits):
        loss = torch.relu(1 + fake_logits).mean()
        loss = loss + torch.relu(1 - real_logits).mean()
        return loss


def get_loss(loss_name):
    if loss_name == LossType.MINIMAX:
        return MinimaxLoss()
    elif loss_name == LossType.WASSERSTEIN:
        return WassersteinLoss()
    elif loss_name == LossType.HINGE:
        return HingeLoss()
    else:
        raise ValueError(f'Unknown loss: {loss_name}')
