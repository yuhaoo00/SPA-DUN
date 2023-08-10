import torch
import torch.nn as nn
from thop import profile
from network.modules import *
from network.unet import *
            
class VCS_UF(nn.Module):
    def __init__(self, num_stage=4, _lambda=1, color_channel=1, CR_channel=8, 
                 width=32, num_blocks=[4,6,8], width_ratio=1, 
                 shortcut='weight', Mask_info=True, CR_info=True,
                 losstype='RMSE', num_loss=3, weight_loss=0.5):
        super(VCS_UF, self).__init__()

        self.num_stage = num_stage
        self._lambda = _lambda
        self.color = color_channel

        self.Unets = nn.ModuleList([ Unet(color_channel, CR_channel, width, num_blocks, width_ratio, shortcut, Mask_info, CR_info) for p in range(self.num_stage) ])

        self.Gammas = nn.Parameter(torch.zeros((self.num_stage)), requires_grad=True)
        
        if losstype == 'MSE':
            self.criterion = nn.MSELoss()
        elif losstype == 'RMSE':
            self.criterion = RMSE()
        elif losstype == 'L1':
            self.criterion = nn.L1Loss()

        self.num_loss = num_loss
        self.weight_loss = weight_loss

    def forward(self, x, Phi, rgb2bayer=1):
        '''
        x    [B, CR, C, H, W]
        Phi  [1, CR,    H, W]
        rgb2bayer [1, 1, C, H, W]
        '''
        y = (x*rgb2bayer).sum(2) # [B, CR, H, W]
        y = (y*Phi).sum(1, keepdim=True) # [B, 1, H, W]
        
        x_list = self.forward_main(y, Phi, rgb2bayer)
        loss = self.loss(x_list, torch.flatten(x,1,2))
        return loss
    
    def forward_main(self, y, Phi, rgb2bayer=1):
        x_list = []
        B, _, H, W = y.shape
        CR = Phi.shape[1]
        Phi_s = torch.sum(Phi, 1, keepdim=True)
        Phi_s[Phi_s==0] = 1
        # y     [B, 1, H, W]
        # Phi   [1, CR, H, W]
        # Phi_s [1, 1, H, W]

        if self.color==3:
            theta = At(y, Phi)     # [B, CR, H, W]
            b = torch.zeros_like(theta)  # [B, CR, H, W]

            x_c = torch.zeros(B, CR*3, H, W).to(y.device) # [B, CR*3, H, W]
            x_c = Bayer2RGB(x_c, theta)

            for i in range(self.num_stage):
                yb = A(theta+b, Phi)     # [B, 1, H, W]
                x_ = theta+b + self._lambda*At(torch.div(y-yb, Phi_s+self.Gammas[i]), Phi) # [B, CR, H, W]
                x_c = Bayer2RGB(x_c, x_-b) # [B, CR*3, H, W]
                x_c = self.Unets[i](x_c, y, Phi)
                #x_c = torch.clamp(x_c, min=0, max=1)
                x_list.append(x_c)
                theta = (x_c.reshape(B,CR,3,H,W)*rgb2bayer).sum(2) # [B, CR, H, W]
                b = b - (x_-theta)
            
        else:
            theta = At(y, Phi)
            b = torch.zeros_like(theta)
            for i in range(self.num_stage):
                yb = A(theta+b, Phi)
                x_ = theta+b + self._lambda*At(torch.div(y-yb, Phi_s+self.Gammas[i]), Phi)
                theta = self.Unets[i](x_-b, y, Phi)
                #theta = torch.clamp(theta, min=0, max=1)
                x_list.append(theta)
                b = b - (x_-theta)

        return x_list

    
    def loss(self, pre, gt):
        loss = self.criterion(pre[-1], gt)
        for i in range(max(len(pre)-self.num_loss, 0), len(pre)-1):
            loss = loss + self.weight_loss*self.criterion(pre[i], gt)
        return loss
    
    def stat_params(self, input_size=[256, 256, 8], device='cpu', log=None):
        x = torch.randn((1, input_size[2], self.color, input_size[0], input_size[1])).to(device)
        phi = torch.randn((1, input_size[2], input_size[0], input_size[1])).to(device)
        if self.color > 1:
            rgb2bayer = torch.randn((1, 1, 3, input_size[0], input_size[1])).to(device)
        else:
            rgb2bayer = 1
        
        flops0, params0 = profile(self, inputs=(x, phi, rgb2bayer), verbose=False)
        txt = '[NET_SIZE] FLOPs:{:>6.2f}G | Params:{:>5.2f}M'.format(flops0/1e9, params0/1e6)
        if log is not None:
            log.info(txt)
        else:
            print(txt)