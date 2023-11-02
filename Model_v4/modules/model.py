import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .unet import Unet
from .utils import *


class CSmodel(nn.Module):
    def __init__(self, color_channel=1, mask_channel=8,
                width=64, width_ratio=1, num_blocks=[4,6,4],
                shortcut='normal', losstype='RMSE',
                Mask_guide=None, Temb_guide=False, temb_channel=64):
        super().__init__()

        self.color = color_channel
        self.CR = mask_channel

        self.Unet = Unet(
            color_channel=color_channel,
            mask_channel=mask_channel,
            temb_channel=temb_channel,
            width=width,
            num_blocks=num_blocks, 
            width_ratio=width_ratio, 
            shortcut=shortcut,
            Mask_guide=Mask_guide, 
            Temb_guide=Temb_guide
        )

        self.Mask_guide = Mask_guide
        self.Temb_guide = Temb_guide

        self.time_encoder = nn.Sequential(
            Timesteps(temb_channel//4, True, 0),
            TimestepEmbedding(temb_channel//4, temb_channel)
         )

        if losstype == 'MSE':
            self.criterion = nn.MSELoss()
        elif losstype == 'RMSE':
            self.criterion = RMSE()
        elif losstype == 'L1':
            self.criterion = nn.L1Loss()

    def get_temb(self, B, T, device):
        timesteps = T * torch.ones((B), device=device) # [B]

        temb = self.time_encoder(timesteps) # [B, Dt]
        return temb # [B, Dt]

    def forward_train(self, x, Phi, rgb2bayer=1):
        '''
        x    [B, CR, C, H, W]
        Phi  [1, CR,    H, W]
        rgb2bayer [1, 1, C, H, W]
        '''
        y = (x*rgb2bayer).sum(2) # [B, CR, H, W]
        y = (y*Phi).sum(1, keepdim=True) # [B, 1, H, W]
        
        x_pred = self.forward(y, Phi)
        loss = self.criterion(x_pred, torch.flatten(x,1,2))
        return loss
    
    def forward(self, y, Phi):
        # y     [B, 1, H, W]
        # Phi   [1, CR, H, W]

        B = y.shape[0]
        _, T, H, W = Phi.shape

        Phi_s = torch.sum(Phi, 1, keepdim=True)
        Phi_s[Phi_s==0] = 1
        y_norm = torch.div(y, Phi_s)
        temb = self.get_temb(B, T, Phi.device)

        x = self.init_x0(y, Phi)
        # x     [B, CR*C, H, W]
        # Phi   [1, CR, H, W]
        
        x_, Phi_, index = self.adaptive_input(x, Phi)
        # x_    [B, CRmax*C, H, W]
        # Phi_  [1, CRmax, H, W]

        x_ = torch.cat([x_, y_norm], dim=1)
        if self.Mask_guide == 'McatR':
            ratio = torch.tensor([T/self.CR]).to(Phi_.device).expand((1, 1, H, W))
            Phi_ = torch.cat([Phi_, ratio], dim=1)
        elif self.Mask_guide == 'McatY':
            Phi_ = torch.cat([Phi_, y_norm], dim=1)

        x_ = self.Unet(x_, Phi_, temb)
        # [B, CRmax*C, H, W]

        x = self.adaptive_output(x_, index, T)
        # [B, CR*C, H, W]

        return x
    
    def init_x0(self, y, Phi):
        B = y.shape[0]
        _, CR, H, W = Phi.shape

        if self.color==3:
            x0_bayer = At(y, Phi)     # [B, CR, H, W]
            x0 = torch.zeros(B, CR*3, H, W).to(y.device) # [B, CR*3, H, W]
            x0 = Bayer2RGB(x0, x0_bayer)  # [B, CR*3, H, W]
        else:
            x0 = At(y, Phi) # [B, CR, H, W]
        return x0
    
    def adaptive_input(self, v, Phi):
        T = Phi.shape[1]
        index = None

        if T < self.CR:
            v = rearrange(v, 'b (t c) h w -> b t c h w', t=T)

            v_tmp = v
            Phi_tmp = Phi
            count = self.CR // T
            remain = self.CR % T
            index = [list(range(T))]
            for i in range(1, count):
                v = torch.flip(v, dims=[1])
                Phi = torch.flip(Phi, dims=[1])
                v_tmp = torch.cat((v_tmp,v), dim=1)
                Phi_tmp = torch.cat((Phi_tmp,Phi), dim=1)
                if i%2:
                    index.append(list(range(T*(i+1)-1, T*i-1, -1)))
                else:
                    index.append(list(range(T*i, T*(i+1))))
            if remain:
                i = count
                v = torch.flip(v, dims=[1])
                Phi = torch.flip(Phi, dims=[1])
                v_tmp = torch.cat((v_tmp,v[:,:remain,:,:,:]), dim=1)
                Phi_tmp = torch.cat((Phi_tmp,Phi[:,:remain,:,:]), dim=1)
                if count%2:
                    index.append(([-1]*(T-remain))+list(range(T*i+remain-1, T*i-1, -1)))
                else:
                    index.append(list(range(T*i, T*i+remain))+([-1]*(T-remain)))

            v = v_tmp.flatten(1,2) # [B, CRmax*C, H, W]
            Phi = Phi_tmp # [1, CRmax, H, W]


        return v, Phi, index

    def adaptive_output(self, v, index, T):
        v = rearrange(v, 'b (t c) h w -> b t c h w', t=self.CR)
        out = v

        if T < self.CR:
            for i in range(T):
                z = v[:,i,:,:,:]
                flag = 1
                for j in range(1, len(index)):
                    if index[j][i] >= 0:
                        z += v[:,index[j][i],:,:,:]
                        flag += 1
                z = (z/flag).unsqueeze(1)

                if i == 0:
                    out = z
                else:
                    out = torch.cat((out,z),1)

        return out.flatten(1,2)