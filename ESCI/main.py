import os
import time
import random
import logging
import datetime
import numpy as np
import scipy.io as sio
import torch
import contextlib
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils import save_img, save_gif, time2file_name, set_seed, print_args
from model import EfficientSCI
from dataset_aug import *

class Handler(object):
    def __init__(self, args):
        if args.train.seed > 0:
            set_seed(args.train.seed)

        if (not args.useCPU) and torch.cuda.is_available():
            self.device = 'cuda'
            self.cuda = True
        else:
            self.device = 'cpu'
            self.cuda = False

        self.args = args
        self.make_dirs()
        self.set_logger()

        self.cr_now = args.flex.cr_train[0]
        self.valid_datapath = self.args.train.valid_datapath

        self.model = EfficientSCI(
                args.model.in_ch, args.model.units,
                args.model.group_num, args.model.color).to(self.device)

        self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 'initial_lr': args.train.lr}], lr=args.train.lr, betas=(0.9, 0.9), weight_decay=0)
        
        self.start_epoch, self.best_epoch_psnr = self.load_ckpt()
        if 'LR_DECAY' in args.train.__dict__:
            lr_step = args.train.LR_DECAY.step
            lr_gamma = args.train.LR_DECAY.gamma
            if len(lr_step) == 1:
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step[0], gamma=lr_gamma, last_epoch=self.start_epoch)
            else:
                steps = list(range(lr_step[0], args.train.epochs, lr_step[1]))
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps, gamma=lr_gamma, last_epoch=self.start_epoch)

        #self.mask = torch.from_numpy(np.float32(sio.loadmat(self.maskpath)['mask'])).to(self.device).permute(2,0,1).unsqueeze(0)
        # [1, cr, H, W] gpu

    def load_ckpt(self):
        start_epoch = 0
        best_psnr = 0
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                ckpt = torch.load(self.args.checkpoint, map_location='cpu') if not self.cuda else torch.load(self.args.checkpoint)
                self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch']+1
                best_psnr = ckpt['best_psnr']
                self.logger.info('{} [Start Here]'.format(self.args.checkpoint))
            else:
                self.logger.error('[Checkpoint Path Does Not Exist!]')
                raise
        return start_epoch, best_psnr

    def train(self):
        # LOAD CKPT to self.model, OPT
        self.model.train()
        best_epoch_psnr = 0

        # Build Dataloader
        train_dataloader = DataLoader(
                dataset=Imgdataset(crop_size=self.args.train.patch_size, color=self.args.model.color, length=max(self.args.flex.cr_train), skip=self.args.flex.cr_skip, root_path=self.args.train.datapath), 
                batch_size=self.args.train.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Print INFO
        self.logger.info('< LR_start:{:.2e}  Batch_num:({}*{})  Img_size:{} >'.format(
                        self.optimizer.param_groups[0]['lr'], self.args.train.batch_size, len(train_dataloader), self.args.train.patch_size))
        if self.args.useAMP: scaler = torch.cuda.amp.GradScaler()
        
        # Prepare RGB2Bayer Matrix
        if self.args.model.color > 1:
            r = np.array([[1, 0], [0, 0]])
            g1 = np.array([[0, 1], [0, 0]])
            g2 = np.array([[0, 0], [1, 0]])
            b = np.array([[0, 0], [0, 1]])

            rgb2bayer = np.zeros([3, self.args.train.patch_size, self.args.train.patch_size])
            rgb2bayer[0, :, :] = np.tile(r, (self.args.train.patch_size//2, self.args.train.patch_size//2))
            rgb2bayer[1, :, :] = np.tile(g1, (self.args.train.patch_size//2, self.args.train.patch_size//2)) + np.tile(g2, (self.args.train.patch_size//2, self.args.train.patch_size//2))
            rgb2bayer[2, :, :] = np.tile(b, (self.args.train.patch_size//2, self.args.train.patch_size//2))
            rgb2bayer = torch.from_numpy(rgb2bayer).to(self.device).unsqueeze(0).unsqueeze(0).type(torch.float32)
            # [1, 1, 3, patch_size, patch_size]
        else:
            rgb2bayer = 1

        ##########################ITER##################################
        for e in range(self.start_epoch, self.args.train.epochs):
            epoch_loss = 0
            time1 = time.time()
            for iteration, data in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                
                self.cr_now = self.args.flex.cr_train[random.randint(0, len(self.args.flex.cr_train)-1)]
                data = normalize_augment(data, self.cuda, self.cr_now, random.randint(0, data.shape[1]-self.cr_now))
                mask = torch.randint(0,2,(data.shape[0], data.shape[1], data.shape[3], data.shape[4]), device=self.device).float()

                # data [bs, cr, C, patch_size, patch_size] gpu
                # mask [bs, cr, patch_size, patch_size]    gpu
                if self.args.useAMP:
                    with torch.cuda.amp.autocast():
                        Loss = self.model.forward_train(data, mask, rgb2bayer)
                    epoch_loss += Loss.data
                    scaler.scale(Loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    Loss = self.model.forward_train(data, mask, rgb2bayer)
                    epoch_loss += Loss.data
                    Loss.backward()
                    self.optimizer.step()
                

            time2 = time.time()
            epoch_loss = epoch_loss/len(train_dataloader)
            self.logger.info('   ===> Epoch:{}/{} Loss:{:2.4f} UseTime:{:.2f}s LR:{:.2e}'.format(
                                e+1, self.args.train.epochs, epoch_loss, time2-time1, self.optimizer.param_groups[0]['lr']))
            
            if e == self.start_epoch+1:
                totaltime = (time2-time1)*(self.args.train.epochs-e-1)
                targetime = time.strftime('%m/%d %H:%M:%S', time.localtime(time1+totaltime))
                self.logger.info('        Expected to take {:2d}h{:2d}m{:2d}s, that is {}'.format(
                               int(totaltime//3600), int((totaltime%3600)//60), int((totaltime%3600)%60), targetime))

            if 'LR_DECAY' in self.args.train.__dict__:
                self.scheduler.step()

            if ((e+1)%self.args.train.ckpt_size==0):
                epoch_psnr = 0
                for cri in range(len(self.args.flex.cr_train)):
                    self.cr_now = self.args.flex.cr_train[cri]
                    epoch_psnr += self.valid()
                epoch_psnr = epoch_psnr/len(self.args.flex.cr_train)
                if epoch_psnr > best_epoch_psnr:
                    best_epoch_psnr = epoch_psnr
                    prefix = 'best'
                else:
                    prefix = 'last'
                
                checkpoint = {'model_state_dict': self.model.state_dict(),
                              'optimizer_state_dict': self.optimizer.state_dict(),
                              'scheduler_state_dict': self.scheduler.state_dict() if ('LR_DECAY' in self.args.train.__dict__) else 0,
                              'epoch': e, 
                              'loss': epoch_loss,
                              'psnr': epoch_psnr,
                              'best_psnr': best_epoch_psnr}
                with contextlib.suppress(OSError):
                    torch.save(checkpoint, '{}/ckpt_{}.pkl'.format(self.ckpt_dir, prefix))

    def valid(self):
        self.logger.info('====================================================================')
        self.valid_list = os.listdir(self.valid_datapath)
        self.pred_sample = []
        psnr_sample = []
        ssim_sample = []
        time_sample = []
        nframe_all = 0

        for i in range(len(self.valid_list)):
            pic = sio.loadmat(self.valid_datapath+'/'+self.valid_list[i])
            orig = np.float32(pic['orig'])/255.
            nframe, H, W, C = orig.shape
            nframe_can = int((nframe//self.cr_now)*self.cr_now)
            orig = orig[:nframe_can, :, :, :]
            mask = torch.randint(0,2,(nframe//self.cr_now, self.cr_now, H, W), device=self.device).float()

            if C > 1:
                r = np.array([[1, 0], [0, 0]])
                g1 = np.array([[0, 1], [0, 0]])
                g2 = np.array([[0, 0], [1, 0]])
                b = np.array([[0, 0], [0, 1]])
                rgb2bayer = np.zeros([3, H, W])
                rgb2bayer[0, :, :] = np.tile(r, (H//2, W//2))
                rgb2bayer[1, :, :] = np.tile(g1, (H//2, W//2)) + np.tile(g2, (H//2, W//2))
                rgb2bayer[2, :, :] = np.tile(b, (H//2, W//2))
                rgb2bayer = torch.from_numpy(rgb2bayer).to(self.device).unsqueeze(0).unsqueeze(0).type(torch.float32)
                # [1, 1, 3, H, W] gpu
                meas = (torch.from_numpy(orig).reshape(-1,self.cr_now,H,W,C).permute(0,1,4,2,3).to(self.device)*rgb2bayer).sum(2) # [*, cr, H, W]
                meas = (meas*mask).sum(1, keepdim=True)
            else:
                rgb2bayer = 1
                meas = (torch.from_numpy(orig).reshape(-1,self.cr_now,H,W).to(self.device)*mask).sum(1, keepdim=True)

            # orig [nframe_can, H, W, C] numpy
            # meas [nmeas, 1, H, W] gpu
            
            time1 = time.time()
            with torch.no_grad():
                out_pic = self.model(meas, mask)
                out_pic = out_pic.reshape(-1,self.cr_now,C,H,W).flatten(0,1).permute(0,2,3,1).cpu().numpy()
                # out [nframe_can, H, W, C] numpy
                psnr_ = []
                ssim_ = []
                for ii in range(nframe_can):
                    out_pic_p = out_pic[ii, ...]
                    orig_p = orig[ii, ...]

                    psnr_.append(compare_psnr(orig_p, out_pic_p, data_range=1.))
                    ssim_.append(compare_ssim(orig_p, out_pic_p, data_range=1., channel_axis=2))

            time2 = time.time()
            usetime = time2-time1

            psnr_sample.append(np.mean(psnr_))
            ssim_sample.append(np.mean(ssim_))
            time_sample.append(usetime)
            self.pred_sample.append(out_pic)
            nframe_all += nframe_can

            self.logger.info('[{}] Avg.PSNR {:.4f}, Avg.SSIM {:.4f}, UseTime {:.4f} s'.format(
                self.valid_list[i], np.mean(psnr_), np.mean(ssim_), usetime))

        self.logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        time_per_meas = np.sum(time_sample)/(nframe_all//self.cr_now)
        self.logger.info('[Overall] CR{} Avg.PSNR {:.4f}, Avg.SSIM {:.4f}, Avg.time {:.4f} s/meas.'.format(
                self.cr_now, np.mean(psnr_sample), np.mean(ssim_sample), time_per_meas))
        self.logger.info('====================================================================')
        return np.mean(psnr_sample)

    def set_logger(self, save=True):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s ||    %(message)s', datefmt='%m/%d %H:%M:%S')

        if save:
            fh = logging.FileHandler('{}/log.txt'.format(self.ckpt_dir))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        print_args(self.args, self.logger)
        self.logger.info('[Mode] {}'.format('GPU' if self.cuda else 'CPU'))
        self.logger.info('[Method] {}'.format(self.args.model.name.upper()))
        self.logger.info('[Checkpoint path] {}'.format(self.ckpt_dir))

    def make_dirs(self):
        self.date_time = str(datetime.datetime.now())
        self.date_time = time2file_name(self.date_time)
        
        self.ckpt_dir = 'Checkpoints/{}/{}'.format(self.args.model.name, self.date_time)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        os.system('cp ESCI/configs/{} {}/configs.yml'.format(self.args.config, self.ckpt_dir))


