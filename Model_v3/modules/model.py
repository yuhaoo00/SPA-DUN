import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .unet import Unet
from .utils import *

class CSmodel(nn.Module):
    def __init__(self, color_channel=1, CR_channel=8, CR_skip_step=4,
                width=64, num_blocks=[4,6,4], width_ratio=1,
                shortcut='normal', block_type='etc', losstype='RMSE',
                Mask_info=False, CR_info=False):
        super().__init__()

        self.color = color_channel
        self.CR = CR_channel
        self.CR_skip_step = CR_skip_step
        self.CR_info = CR_info

        self.Unet = Unet(
            in_channel=(CR_channel*color_channel)+1,
            out_channel=CR_channel*color_channel,
            mask_channel=CR_channel,
            width=width,
            num_blocks=num_blocks, 
            width_ratio=width_ratio, 
            shortcut=shortcut,
            block_type=block_type, 
            Mask_info=Mask_info, 
            CR_info=CR_info
        )

        if losstype == 'MSE':
            self.criterion = nn.MSELoss()
        elif losstype == 'RMSE':
            self.criterion = RMSE()
        elif losstype == 'L1':
            self.criterion = nn.L1Loss()

    def forward_train(self, x, Phi, rgb2bayer=1):
        '''
        x    [B, CR, C, H, W]
        Phi  [B, CR,    H, W]
        rgb2bayer [1, 1, C, H, W]
        '''
        y = (x*rgb2bayer).sum(2) # [B, CR, H, W]
        y = (y*Phi).sum(1, keepdim=True) # [B, 1, H, W]
        
        x_pred = self.forward(y, Phi)
        loss = self.criterion(x_pred, torch.flatten(x,1,2))
        return loss
    
    def forward(self, y, Phi):
        B, _, H, W = y.shape
        CR = Phi.shape[1]
        Phi_s = torch.sum(Phi, 1, keepdim=True)
        Phi_s[Phi_s==0] = 1
        y_norm = torch.div(y, Phi_s)
        # y     [B, 1, H, W]
        # Phi   [B, CR, H, W]

        if self.color==3:
            x0_bayer = At(y, Phi)     # [B, CR, H, W]
            x0 = torch.zeros(B, CR*3, H, W).to(y.device) # [B, CR*3, H, W]
            x0 = Bayer2RGB(x0, x0_bayer)  # [B, CR*3, H, W]
        else:
            x0 = At(y, Phi) # [B, CR, H, W]
        x_pred = torch.zeros_like(x0)

        for i in range(0, CR+1, self.CR_skip_step):
            if (i + self.CR) > CR:
                break
            xi = x0[:,i*self.color:(i+self.CR)*self.color,:,:]
            xi = torch.cat((xi, y_norm), dim=1)
            mi = Phi[:,i*self.color:(i+self.CR)*self.color,:,:]
            if self.CR_info:
                mi = torch.cat((mi, y_norm), dim=1)
            xi = self.Unet(xi, mi)
            x_pred[:,i*self.color:(i+self.CR)*self.color,:,:] += xi/2
        
        x_pred[:,:(self.CR_skip_step)*self.color,:,:] *= 2
        x_pred[:,-1*(self.CR_skip_step)*self.color:,:,:] *= 2

        return x_pred