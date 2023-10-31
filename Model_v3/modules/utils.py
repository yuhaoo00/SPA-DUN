import torch
import torch.nn as nn

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