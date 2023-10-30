from modules.Unet import UNet2DModel
from thop import profile
import torch
model = UNet2DModel(
    in_channels=8,
    out_channels=8,
    cross_attention_dim=16,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).cuda()
print(model)

x = torch.ones((1,8,256,256)).cuda()
timestep = torch.ones((1)).cuda()
encoder_hidden_states = torch.ones((4096,8,16)).cuda()
flops, params = profile(model, (x, timestep, encoder_hidden_states))
print(flops/1e9, params/1e6)