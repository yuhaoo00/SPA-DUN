import torch
import os
import time
import yaml
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import scipy.io as sio
from network.unfolding import VCS_UF
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils import save_img, save_gif, dict2namespace

parser = ArgumentParser()
parser.add_argument('--useCPU', action='store_true', default=False)
parser.add_argument('--real',  action='store_true', default=False)
parser.add_argument('--output',  action='store_true', default=False)
parser.add_argument('--slice',  action='store_true', default=False)

parser.add_argument('--dir', type=str, default='Checkpoints/SPA-DUN-simu')
parser.add_argument('--modelpath', type=str, default='ckpt_best.pkl')
parser.add_argument('--CR', type=int, default=8)
parser.add_argument('--datapath', type=str, default='Dataset/Simu_test/gray/256')
parser.add_argument('--maskpath', type=str, default='Dataset/Masks/new/rand_cr50.mat')

args = parser.parse_args()

with open('{}/configs.yml'.format(args.dir), 'r') as f:
    configs1 = yaml.safe_load(f)
args = dict2namespace(configs1, args)
args.modelpath = '{}/{}'.format(args.dir, args.modelpath)
res_dir = args.dir.replace('Checkpoints', 'Outputs')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if (not args.useCPU) and torch.cuda.is_available():
    device = 'cuda'
    cuda = True
else:
    device = 'cpu'
    cuda = False

model = VCS_UF(args.model.num_stage, 1, args.model.color, args.flex.cr_model,
               args.model.width, args.model.num_blocks, args.model.width_ratio,
               args.model.shortcut, args.model.Mask_info, args.model.CR_info,
               args.model.losstype, args.model.num_loss, args.model.weight_loss).to(device)

if os.path.exists(args.modelpath):
    ckpt = torch.load(args.modelpath, map_location='cpu') if not cuda else torch.load(args.modelpath)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
else:
    print("EMPTY MODEL")

if not args.real:
    mask = torch.from_numpy(np.float32(sio.loadmat(args.maskpath)['mask'])).to(device).permute(2,0,1).unsqueeze(0)
    print('[Use_mask] {}'.format(args.maskpath))
    # mask [1, CR, H, W] gpu

    test_list = os.listdir(args.datapath)
    psnr_sample = []
    ssim_sample = []
    time_sample = []
    pred_sample = []
    nmeas_all = 0
    
    pd_log = {}

    for i in range(len(test_list)):
        orig = (np.float32(sio.loadmat(args.datapath+'/'+test_list[i])['orig'])/255.)
        nframe, H, W, C = orig.shape
        nframe_can = int((nframe//args.CR)*args.CR)
        orig = orig[:nframe_can, :, :, :]

        if i == 0:
            mask = mask[:, :args.CR, :H, :W]
            model.stat_params(input_size=[H, W, args.CR], device=device)

        if C > 1:
            r = np.array([[1, 0], [0, 0]])
            g1 = np.array([[0, 1], [0, 0]])
            g2 = np.array([[0, 0], [1, 0]])
            b = np.array([[0, 0], [0, 1]])
            rgb2bayer = np.zeros([3, H, W])
            rgb2bayer[0, :, :] = np.tile(r, (H//2, W//2))
            rgb2bayer[1, :, :] = np.tile(g1, (H//2, W//2)) + np.tile(g2, (H//2, W//2))
            rgb2bayer[2, :, :] = np.tile(b, (H//2, W//2))
            orig_bayer = np.sum(orig*(rgb2bayer.transpose(1,2,0)[np.newaxis,:,:,:]), axis=3)
            # orig_bayer [nframe_can, H, W] gpu
            rgb2bayer = torch.from_numpy(rgb2bayer).to(device).unsqueeze(0).unsqueeze(0).type(torch.float32)
            # rgb2bayer [1, 1, 3, H, W] gpu
            meas = (torch.from_numpy(orig).reshape(-1,args.CR,H,W,C).permute(0,1,4,2,3).to(device)*rgb2bayer).sum(2) # [*, CR, H, W]
            meas = (meas*mask).sum(1, keepdim=True)
        else:
            rgb2bayer = 1
            meas = (torch.from_numpy(orig).reshape(-1,args.CR,H,W).to(device)*mask).sum(1, keepdim=True)

        # orig [nframe_can, H, W, C] numpy
        # meas [nmeas, 1, H, W] gpu

        nmeas = meas.shape[0]
        out_pics = []
        
        time1 = time.time()
        for j in range(nmeas):
            with torch.no_grad():
                out_pic = model.forward_main(meas[j:j+1,...], mask, rgb2bayer)[-1]
                out_pic = out_pic.reshape(-1, C, H, W).permute(0,2,3,1).cpu().numpy()
                # out [CR, H, W, C] numpy
                out_pics.append(out_pic)

        time2 = time.time()
        usetime = time2-time1

        out_pics = np.array(out_pics).reshape(-1, H, W, C)
        # [nframe_can, H, W, C] numpy

        psnr_ = []
        ssim_ = []
        if C > 1:
            rgb2bayer = rgb2bayer[0,0,:,:,:].cpu().numpy().transpose(1,2,0)
            # rgb2bayer [H, W, 3] numpy
        for j in range(nframe_can):
            if C > 1:
                orig_p = orig_bayer[j,...]
                pred_p = np.sum(out_pics[j,...]*rgb2bayer, axis=2)
            else:
                orig_p = orig[j,:,:,0]
                pred_p = out_pics[j,:,:,0]
            psnr_.append(compare_psnr(orig_p, pred_p, data_range=1.))
            ssim_.append(compare_ssim(orig_p, pred_p, data_range=1.))

        psnr_sample.append(np.mean(psnr_))
        ssim_sample.append(np.mean(ssim_))
        time_sample.append(usetime)
        pred_sample.append(out_pics)
        nmeas_all += nmeas

        pd_log[test_list[i][:-4]+"_psnr"] = psnr_
        pd_log[test_list[i][:-4]+"_ssim"] = ssim_

        print("[{}] Avg.PSNR: {:.4f}, Avg.SSIM {:.4f}, UseTime {:.4f} s".format(
            test_list[i], np.mean(psnr_), np.mean(ssim_), usetime))

        if args.output:
            subdir = '{}/cr{}/{}'.format(res_dir, args.CR, test_list[i][:-4])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            video = out_pics*255
            video = video.astype(int)
            video[video>255] = 255 
            video[video<0] = 0
            for f in range(video.shape[0]):
                save_img(video[f,...], '{}/{}.png'.format(subdir, f))
            save_gif(video, '{}/play.gif'.format(subdir))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    time_per_meas = np.sum(time_sample)/nmeas_all
    print("[Overall] Avg.PSNR {:.4f}, Avg.SSIM {:.4f}, Avg.time {:.4f} s/meas.".format(
            np.mean(psnr_sample), np.mean(ssim_sample), time_per_meas))
    print("====================================================================")

    if args.slice:
        pd.DataFrame(pd_log).to_csv('{}/cr{}/slice.csv'.format(res_dir, args.CR))

else:
    mask = torch.from_numpy(np.float32(sio.loadmat(args.maskpath)['mask'])).to(device).permute(2,0,1).unsqueeze(0)
    mask = mask[:, :args.CR, :, :]
    print('[Use_mask] {}'.format(args.maskpath))
    # mask [1, CR, H, W] gpu

    test_list = os.listdir(args.datapath)
    time_sample = []
    pred_sample = []
    nmeas_all = 0

    for i in range(len(test_list)):
        meas = torch.from_numpy(np.float32(sio.loadmat(args.datapath+'/'+test_list[i])['meas']))#*args.CR/(255.*2)
        if len(meas.shape) == 2:
            meas = meas.to(device).unsqueeze(0).unsqueeze(0)
        else:
            meas = meas.permute(2,0,1).to(device).unsqueeze(1)
        # meas [nmeas, 1, H, W] gpu
        nmeas, _, H, W = meas.shape
        out_pics = []
        
        time1 = time.time()
        for j in range(nmeas):
            with torch.no_grad():
                out_pic = model.forward_main(meas[j:j+1,...], mask, 1)[-1]
                out_pic = out_pic.reshape(-1, H, W).cpu().numpy()
                # out [CR, H, W] numpy
                if args.CR == 14 and j%2 != 0: # when LM and j is odd
                    out_pic = out_pic[::-1,:,:]
                out_pics.append(out_pic)

        time2 = time.time()
        usetime = time2-time1

        out_pics = np.array(out_pics).reshape(-1, H, W)
        # [nframe, H, W] numpy

        time_sample.append(usetime)
        pred_sample.append(out_pics)
        nmeas_all += nmeas

        print("[{}] UseTime {:.4f} s".format(test_list[i], usetime))

        if args.output:
            subdir = '{}/cr{}/{}'.format(res_dir, args.CR, test_list[i][:-4])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            video = out_pics*255
            video = video.astype(int)
            video[video>255] = 255 
            video[video<0] = 0
            for f in range(video.shape[0]):
                save_img(video[f,...], '{}/{}.png'.format(subdir, f))
            save_gif(video, '{}/play.gif'.format(subdir))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    time_per_meas = np.sum(time_sample)/nmeas_all
    print("[Overall] Avg.time {:.4f} s/meas.".format(time_per_meas))
    print("====================================================================")
