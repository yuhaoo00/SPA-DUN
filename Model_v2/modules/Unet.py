import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .blocks_down import *
from .blocks_mid import *
from .blocks_up import *

class UNet2DModel(nn.Module):
    def __init__(
        self,
        maxt: int = 8,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types: Tuple[str] = ("CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (320, 640, 1280),
        layers_per_block: int = 1,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        cross_attention_dim: int = 16,
        attention_head_dim: int = 8,
        time_embed_dim: int = 1280,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                maxt=maxt,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                cross_attention_dim=cross_attention_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = MidBlock2DCrossAttn(
            maxt=maxt,
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                maxt=maxt,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                cross_attention_dim=cross_attention_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.FloatTensor,
        temb: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # 1. Time
        temb = temb.reshape(x.shape[0],-1)

        # 2. pre-process
        x = self.conv_in(x)

        # 3. down
        down_block_res_samples = (x,)
        for downsample_block in self.down_blocks:
            x, res_samples = downsample_block(hidden_states=x, temb=temb, encoder_hidden_states=encoder_hidden_states)
            down_block_res_samples += res_samples

        # 4. mid
        x = self.mid_block(x, temb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            x = upsample_block(x, res_samples, temb, encoder_hidden_states=encoder_hidden_states)

        # 6. post-process
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x