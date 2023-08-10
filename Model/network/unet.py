import torch
import torch.nn as nn
from thop import profile
from network.modules import *


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

class ResBlocks(nn.Module):
    def __init__(self, num_block, chan, chan_MI, width_ratio=1, shortcut='normal', MI=False):
        super(ResBlocks, self).__init__()
        self.blocks = nn.ModuleList([])
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
    def __init__(self, color_channel, CR_channel, width,
                    num_blocks=[], width_ratio=1, shortcut='normal', Mask_info=False, CR_info=False):
        super().__init__()
        self.intro = nn.Conv2d(color_channel*CR_channel+1, width, kernel_size=3, stride=1, padding=1)
        self.ending = nn.Conv2d(width, color_channel*CR_channel, kernel_size=3, stride=1, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.maskdowns = nn.ModuleList()
        self.refinement = nn.ModuleList()

        self.color = color_channel
        self.Mask_info = Mask_info
        self.CR_info = CR_info
        self.CR_m = CR_channel
        chan_MI = CR_channel+1 if self.CR_info else CR_channel
        chan = width

        for i in range(len(num_blocks)-1):
            self.encoders.append(ResBlocks(num_blocks[i], chan, chan_MI, width_ratio, shortcut, Mask_info))
            self.downs.append(Downsample(chan))
            self.maskdowns.append(Downsample(chan_MI))
            chan = chan * 2
            chan_MI = chan_MI * 2

        self.middle_blks = ResBlocks(num_blocks[-1], chan, chan_MI, width_ratio, shortcut, Mask_info)

        for i in range(len(num_blocks)-2, -1, -1):
            self.ups.append(Upsample(chan))
            chan = chan // 2
            chan_MI = chan_MI // 2
            self.decoders.append(ResBlocks(num_blocks[i], chan, chan_MI, width_ratio, shortcut, Mask_info))

    def forward_main(self, v, phi):
        # v    [B, CR_m*C+1, H, W]
        # phi  [1, CR_m+1,   H, W]
        x = self.intro(v) # [B, C_width, H, W]
        encs = []
        phis = []
        for encoder, down, maskdown in zip(self.encoders, self.downs, self.maskdowns):
            x = encoder(x, phi)
            encs.append(x)
            phis.append(phi)
            x = down(x)
            phi = maskdown(phi)

        x = self.middle_blks(x, phi)

        for decoder, up, enc_skip, phi_skip in zip(self.decoders, self.ups, encs[::-1], phis[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, phi_skip)

        # [B, C_width, H, W]
        x = self.ending(x)
        # [B, CR_m*C, H, W]
        return x

    def forward(self, v, y, phi):
        # v     [B, D, H, W]
        # y     [B, 1, H, W]
        # phi   [1, CR,H, W]

        B, D, H, W = v.shape
        CR = phi.shape[1]
        in_ch = self.CR_m*self.color
        phi_s = torch.sum(phi, 1, keepdim=True)
        phi_s[phi_s==0] = 1
        im = v

        if D == in_ch:
            ########## Prepare V ############
            y = torch.div(y, phi_s)
            v = torch.cat((v,y), dim=1)
            ########## Prepare Phi ############
            if self.CR_info:
                cr_info = torch.tensor([CR/self.CR_m]).to(phi.device).expand((1, 1, H, W))
                phi = torch.cat((phi, cr_info), dim=1) # [1, CR_m+1, H, W]
            return self.forward_main(v, phi) + im
        
        elif D < in_ch:
            ########## Prepare V & Phi ############
            v = v.reshape(B, CR, self.color, H, W)
            v_tmp = v
            phi_tmp = phi
            count = self.CR_m // CR
            remain = self.CR_m % CR
            index = [list(range(CR))]
            for i in range(1, count):
                v = torch.flip(v, dims=[1])
                phi = torch.flip(phi, dims=[1])
                v_tmp = torch.cat((v_tmp,v), dim=1)
                phi_tmp = torch.cat((phi_tmp,phi), dim=1)
                if i%2:
                    index.append(list(range(CR*(i+1)-1, CR*i-1, -1)))
                else:
                    index.append(list(range(CR*i, CR*(i+1))))
            if remain:
                i = count
                v = torch.flip(v, dims=[1])
                phi = torch.flip(phi, dims=[1])
                v_tmp = torch.cat((v_tmp,v[:,:remain,:,:,:]), dim=1)
                phi_tmp = torch.cat((phi_tmp,phi[:,:remain,:,:]), dim=1)
                if count%2:
                    index.append(([-1]*(CR-remain))+list(range(CR*i+remain-1, CR*i-1, -1)))
                else:
                    index.append(list(range(CR*i, CR*i+remain))+([-1]*(CR-remain)))

            v_tmp = v_tmp.flatten(1,2) # [B, CR_m*C, H, W]
            y = torch.div(y, phi_s)
            #y = y*(self.CR_m/CR)
            v = torch.cat((v_tmp, y), dim=1) # [B, CR_m*C+1, H, W]
            phi = phi_tmp # [1, CR_m, H, W]
            if self.CR_info:
                cr_info = torch.tensor([CR/self.CR_m]).to(phi.device).expand((1, 1, H, W))
                phi = torch.cat((phi, cr_info), dim=1) # [1, CR_m+1, H, W]

            #################################
            v = self.forward_main(v, phi).reshape(B, -1, self.color, H, W) # [B, CR_m, C, H, W]
            ########## Arrange X ##########
            for i in range(CR):
                z = v[:,i,:,:,:]
                flag = 1
                for j in range(1, len(index)):
                    if index[j][i] >= 0:
                        z += v[:,index[j][i],:,:,:]
                        flag += 1
                z = (z/flag).unsqueeze(1)

                if i == 0:
                    x = z
                else:
                    x = torch.cat((x,z),1)
                    
            return x.flatten(1,2) + im
        
        else:
            v = v.reshape(B, CR, self.color, H, W)
            y = torch.div(y, phi_s)
            count = CR // self.CR_m
            remain = CR % self.CR_m

            for i in range(count):
                v_ = v[:,i*self.CR_m:(i+1)*self.CR_m,:,:,:].flatten(1,2)
                v_ = torch.cat((v_, y), dim=1) # [B, CR_m*C+1, H, W]
                phi_ = phi[:,i*self.CR_m:(i+1)*self.CR_m,:,:] # [1, CR_m, H, W]
                if self.CR_info:
                    cr_info = torch.tensor([CR/self.CR_m]).to(phi.device).expand((1, 1, H, W))
                    phi_ = torch.cat((phi_, cr_info), dim=1) # [1, CR_m+1, H, W]
                v_ = self.forward_main(v_, phi_).reshape(B, -1, self.color, H, W)  # [B, CR_m, C, H, W]
                if i == 0:
                    x = v_
                else:
                    x = torch.cat((x, v_), dim=1)
            
            if remain:
                v_ = v[:,-1*self.CR_m:,:,:,:].flatten(1,2)
                v_ = torch.cat((v_, y), dim=1) # [B, CR_m*C+1, H, W]
                phi_ = phi[:,-1*self.CR_m:,:,:] # [1, CR_m, H, W]
                if self.CR_info:
                    cr_info = torch.tensor([CR/self.CR_m]).to(phi.device).expand((1, 1, H, W))
                    phi_ = torch.cat((phi_, cr_info), dim=1) # [1, CR_m+1, H, W]
                v_ = self.forward_main(v_, phi_).reshape(B, -1, self.color, H, W)  # [B, CR_m, C, H, W]
                x = torch.cat((x, v_[:,-remain:,:,:,:]), dim=1) # [B, CR, C, H, W]
            
            return x.flatten(1,2) + im