import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .Unet import UNet2DModel
from .embeddings import TimestepEmbedding, Timesteps

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = torch.sqrt(self.mse(pred, target))
        return loss

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

class CSmodel(nn.Module):
    def __init__(self, color_channel=1, CR_channel=8, CR_skip_step=4,
                 down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D"),
                 up_block_types=("CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                 block_out_channels=(320, 640, 1280),
                 attention_head_dim=8,
                 freq_shift=0,
                 flip_sin_to_cos=True,
                 losstype='RMSE'):
        super(CSmodel, self).__init__()

        self.color = color_channel
        self.CR = CR_channel
        self.CR_skip_step = CR_skip_step
        self.unet_scale = 2**(len(down_block_types)-1)
        self.mask_embed_dim = self.unet_scale**2

        time_embed_dim = block_out_channels[0] * 4
        

        self.Unet = UNet2DModel(
            maxt=CR_channel,
            in_channels=(CR_channel*color_channel)+1,
            out_channels=CR_channel*color_channel,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            cross_attention_dim=self.mask_embed_dim,
            attention_head_dim=attention_head_dim,
            time_embed_dim=time_embed_dim,
        )
        self.time_encoder = nn.Sequential(
            Timesteps(block_out_channels[0]//CR_channel, flip_sin_to_cos, freq_shift),
            TimestepEmbedding(block_out_channels[0]//CR_channel, time_embed_dim//CR_channel)
         )
        self.time_for_mask_proj = nn.Linear(time_embed_dim//CR_channel, self.mask_embed_dim, bias=False)
        
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
            ti = self.get_time_embed(i, i+self.CR, B, y.device)
            mi = self.get_mask_embed(ti, Phi[:,i:i+self.CR,:,:])

            
            xi = self.Unet(xi, ti, mi)
            x_pred[:,i*self.color:(i+self.CR)*self.color,:,:] += xi/2
        
        x_pred[:,:(self.CR_skip_step)*self.color,:,:] *= 2
        x_pred[:,-1*(self.CR_skip_step)*self.color:,:,:] *= 2

        return x_pred
    
    def get_time_embed(self, CR_l, CR_r, B, device):
        timesteps = torch.arange(CR_l,CR_r).to(device).float()
        timesteps = timesteps[None,:] * torch.ones((B,CR_r-CR_l), dtype=timesteps.dtype, device=device) # [B, maxCR]

        temb = self.time_encoder(timesteps) # [B, maxCR, time_embed_dim//maxCR]
        return temb 
    
    def get_mask_embed(self, temb, mask):
        B, CR, H, W = mask.shape
        temb = self.time_for_mask_proj(temb) # [B, maxCR, mask_embed_dim]
        mask = mask.reshape(-1, 1, H, W)
        mask_p = F.unfold(mask, (self.unet_scale, self.unet_scale), stride=(self.unet_scale, self.unet_scale))
        mask_p = rearrange(mask_p,"(b t) d p -> b p t d", b=B, t=CR)
        mask_p += temb[:, None, :, :]
        return mask_p.reshape(-1, CR, mask_p.shape[-1]) # [B*p, maxCR, mask_embed_dim]