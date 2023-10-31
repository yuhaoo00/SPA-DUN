import torch
import torch.nn as nn

class ResBlocks(nn.Module):
    def __init__(self, block_type, num_block, chan, chan_MI, width_ratio=1, shortcut='normal', MI=False):
        super(ResBlocks, self).__init__()
        self.blocks = nn.ModuleList([])
        if block_type == 'etc':
            from .etc import ResBlock
        elif block_type == 'restormer':
            from .restormer import ResBlock

        for _ in range(num_block):
            self.blocks.append(ResBlock(chan, chan_MI, width_ratio, shortcut, MI))

    def forward(self, x, mask):
        for layer in self.blocks:
            x = layer(x, mask)
        return x

class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()

        self.body = nn.Conv2d(channel, channel*2, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channel, channel*2, kernel_size=1, stride=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Unet(nn.Module):
    def __init__(self, 
                in_channel, 
                out_channel,
                mask_channel,
                width,
                num_blocks=[], 
                width_ratio=1, 
                shortcut='normal',
                block_type='etc', 
                Mask_info=False, 
                CR_info=False):
        super().__init__()
        self.intro = nn.Conv2d(in_channel, width, kernel_size=3, stride=1, padding=1)
        self.ending = nn.Conv2d(width, out_channel, kernel_size=3, stride=1, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.maskdowns = nn.ModuleList()
        self.refinement = nn.ModuleList()

        self.Mask_info = Mask_info
        self.CR_info = CR_info
        self.CR_m = mask_channel
        chan_MI = mask_channel+1 if self.CR_info else mask_channel
        chan = width

        for i in range(len(num_blocks)-1):
            self.encoders.append(ResBlocks(block_type, num_blocks[i], chan, chan_MI, width_ratio, shortcut, Mask_info))
            self.downs.append(Downsample(chan))
            self.maskdowns.append(Downsample(chan_MI))
            chan = chan * 2
            chan_MI = chan_MI * 2

        self.middle_blks = ResBlocks(block_type, num_blocks[-1], chan, chan_MI, width_ratio, shortcut, Mask_info)

        for i in range(len(num_blocks)-2, -1, -1):
            self.ups.append(Upsample(chan))
            chan = chan // 2
            chan_MI = chan_MI // 2
            self.decoders.append(ResBlocks(block_type, num_blocks[i], chan, chan_MI, width_ratio, shortcut, Mask_info))

    def forward(self, v, phi):
        # v    [B, CR_m*C+1, H, W]
        # phi  [1, CR_m+1,   H, W]
        x = self.intro(v) # [B, C_width, H, W]
        encs = []
        phis = []
        for encoder, down, maskdown in zip(self.encoders, self.downs, self.maskdowns):
            x = encoder(x, phi)
            encs.append(x)
            x = down(x)

            phis.append(phi)
            if self.Mask_info: phi = maskdown(phi)

        x = self.middle_blks(x, phi)

        for decoder, up, enc_skip, phi_skip in zip(self.decoders, self.ups, encs[::-1], phis[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, phi_skip)

        # [B, C_width, H, W]
        x = self.ending(x)
        # [B, CR_m*C, H, W]
        return x