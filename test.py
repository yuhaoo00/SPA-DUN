import torch
ckpt = torch.load('Checkpoints/SPA-DUN-real/ckpt_best.pkl')
print(ckpt['epoch'])