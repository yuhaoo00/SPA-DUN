import torch
import torch.nn as nn
from .utils import *

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class ResBlock(nn.Module):
    def __init__(self, chan, chan_MI, width_ratio=1, shortcut='normal', MI=False):
        super().__init__()
        dw_chan = chan * width_ratio
        self.conv1 = nn.Conv2d(in_channels=chan, out_channels=dw_chan, kernel_size=1, padding=0, stride=1, groups=1, bias=True) # Nochange w/o bias 
        self.conv2 = nn.Conv2d(in_channels=dw_chan, out_channels=dw_chan, kernel_size=3, padding=1, stride=1, groups=dw_chan, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_chan, out_channels=chan, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        if MI:
            self.MI_Attn = MaskGuidedMechanism(chan_MI, chan)
            self.MI_flag = True
        else:
            self.MI_flag = False

        self.gelu = nn.GELU() # Significant than ReLU

        ffn_chan = width_ratio * chan
        self.conv4 = nn.Conv2d(in_channels=chan, out_channels=ffn_chan, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 
        self.conv5 = nn.Conv2d(in_channels=ffn_chan, out_channels=chan, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(chan)
        self.norm2 = LayerNorm2d(chan)

        if shortcut == 'weight':
            self.beta = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)  # Significant than NAN
            self.gamma = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True) # Significant than NAN
        elif shortcut == 'normal':
            self.beta = 1
            self.gamma = 1

    def forward(self, inp, mask):
        # [b, channel, h, w] -> [b, channel, h, w]
        # [b, channel, h, w]
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        if self.MI_flag:
            x = x * self.MI_Attn(mask)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.conv5(x)

        return y + x * self.gamma