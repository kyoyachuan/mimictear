import torch
from torch.nn import functional as F

from .contants import LossType


class MinimaxLoss:
    def g_loss(self, fake_logits):
        return -torch.log(torch.sigmoid(fake_logits)).mean()

    def d_loss(self, real_logits, fake_logits):
        return -(torch.log(torch.sigmoid(real_logits)).mean() + torch.log(1 - torch.sigmoid(fake_logits)).mean())


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
