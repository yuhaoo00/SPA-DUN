import math
import torch
import torch.nn as nn
from typing import Optional

def A(x, Phi):
    '''
    [batch, CR, H, W] x (in Bayer)
    [batch, CR, H, W] Phi
    Return:
    [batch, 1,  H, W] y
    '''
    y = (x*Phi).sum(1, keepdim=True) # [B, 1, H, W]
    return y

def At(y, Phi):
    '''
    [batch, 1,  H, W] y
    [batch, CR, H, W] Phi
    Return:
    [batch, CR, H, W] x (in Bayer)
    '''
    temp = y.repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def Bayer2RGB(x_rgb, x_bayer):
    # x_rgb   [B, CR*3, H, W]
    # x_bayer [B, CR, H, W]
    bayer = [[0,0], [0,1], [1,0], [1,1]]
    x_rgb[:, 0::3, bayer[0][0]::2, bayer[0][1]::2] =  x_bayer[:, :, bayer[0][0]::2, bayer[0][1]::2]
    x_rgb[:, 1::3, bayer[1][0]::2, bayer[1][1]::2] =  x_bayer[:, :, bayer[1][0]::2, bayer[1][1]::2]
    x_rgb[:, 1::3, bayer[2][0]::2, bayer[2][1]::2] =  x_bayer[:, :, bayer[2][0]::2, bayer[2][1]::2]
    x_rgb[:, 2::3, bayer[3][0]::2, bayer[3][1]::2] =  x_bayer[:, :, bayer[3][0]::2, bayer[3][1]::2]
    return x_rgb

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = torch.sqrt(self.mse(pred, target))
        return loss
    
class MaskGuidedMechanism(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2, bias=True, groups=out_ch)

    def forward(self, mask):
        mask = self.conv1(mask)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask)))
        res = mask * attn_map
        mask = mask + res
        return mask
    
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, :, None].float() * emb[None, None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, :, half_dim:], emb[:, :, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "gelu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"{act_fn} does not exist. Make sure to define one of 'silu', 'mish', or 'gelu'")

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        elif post_act_fn == "silu":
            self.post_act = nn.SiLU()
        elif post_act_fn == "mish":
            self.post_act = nn.Mish()
        elif post_act_fn == "gelu":
            self.post_act = nn.GELU()
        else:
            raise ValueError(f"{post_act_fn} does not exist. Make sure to define one of 'silu', 'mish', or 'gelu'")

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb