import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()
    return linear


def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()
    return conv


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.query = init_conv(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.key = init_conv(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.value = init_conv(nn.Conv1d(in_channel, in_channel, 1))

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed_g = init_linear(nn.Linear(n_class, in_channel))
        self.embed_b = init_linear(nn.Linear(n_class, in_channel))

    def forward(self, input, cond):
        out = self.bn(input)
        gamma = self.embed_g(cond)
        beta = self.embed_b(cond)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=4,
        padding=1,
        stride=2,
        deconv=False,
        n_class=None,
        cbn=False,
        activation=leaky_relu,
        self_attention=False
    ):
        super().__init__()

        self.deconv = deconv
        if deconv:
            self.conv = init_conv(nn.ConvTranspose2d(
                in_channel, out_channel, kernel_size, stride, padding
            ))
        else:
            self.conv = init_conv(nn.Conv2d(
                in_channel, out_channel, kernel_size, stride, padding
            ))

        self.activation = activation
        self.cbn = cbn
        if cbn:
            self.norm = ConditionalNorm(out_channel, n_class)
        else:
            self.norm = nn.BatchNorm2d(out_channel)

        self.self_attention = self_attention
        if self_attention:
            self.attention = SelfAttention(out_channel)

    def forward(self, input, cond=None):
        out = input
        out = self.conv(out)

        if self.cbn:
            out = self.norm(out, cond)
        else:
            out = self.norm(out)

        if self.activation is not None:
            out = self.activation(out)

        if self.self_attention:
            out = self.attention(out)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim=128, cond_dim=128, n_class=24, self_attention=False, cbn=False):
        super().__init__()
        self.code_dim = code_dim
        self.cond_dim = cond_dim
        self.self_attention = self_attention
        self.cbn = cbn

        if not cbn:
            self.cond_embed = init_linear(nn.Linear(n_class, cond_dim))
        cond_dim = cond_dim if not cbn else 0

        self.lin_code = init_linear(nn.Linear(code_dim, 1 * 1 * 512))
        self.conv = nn.ModuleList(
            [
                ConvBlock(512 + cond_dim, 512, padding=0, deconv=True, n_class=n_class, cbn=cbn),
                ConvBlock(512, 512, deconv=True, n_class=n_class, cbn=cbn),
                ConvBlock(512, 256, deconv=True, n_class=n_class, cbn=cbn, self_attention=self_attention),
                ConvBlock(256, 128, deconv=True, n_class=n_class, cbn=cbn)
            ]
        )

        self.colorize = init_conv(nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1))

    def forward(self, input, cond=None):
        out = torch.relu(self.lin_code(input))
        out = out.view(-1, 512, 1, 1)

        if not self.cbn:
            out_cond = torch.relu(self.cond_embed(cond))
            out_cond = out_cond.view(-1, self.cond_dim, 1, 1)
            out = torch.cat([out, out_cond], 1)

        for conv in self.conv:
            out = conv(out, cond)

        return torch.tanh(self.colorize(out))


class Discriminator(nn.Module):
    def __init__(self, n_class=24, projection=False, self_attention=False, img_shape=(3, 64, 64)):
        super().__init__()
        self.C, self.H, self.W = img_shape
        self.projection = projection
        self.self_attention = self_attention
        self.input_dim = self.C

        if not projection:
            self.cond_embed = init_linear(nn.Linear(n_class, self.C * self.H * self.W))
            self.input_dim *= 2
        else:
            self.cond_embed = init_linear(nn.Linear(n_class, 512))

        self.conv = nn.Sequential(ConvBlock(self.input_dim, 128),
                                  ConvBlock(128, 256),
                                  ConvBlock(256, 512, self_attention=self_attention),
                                  ConvBlock(512, 512),
                                  ConvBlock(512, 512))

        self.linear = init_linear(nn.Linear(512, 1))

    def forward(self, input, cond):
        if not self.projection:
            out_cond = self.cond_embed(cond)
            out_cond = out_cond.view(-1, self.C, self.H, self.W)
            input = torch.cat([input, out_cond], 1)

        out = self.conv(input)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)

        if self.projection:
            embed = self.cond_embed(cond)
            prod = (out * embed).sum(1)
            out_linear = out_linear + prod

        return out_linear
