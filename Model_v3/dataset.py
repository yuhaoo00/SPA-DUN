import torch
import cv2
import os
import os.path as osp
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class MyRandomResize(v2.Transform):
    def __init__(self, scale=(0.8,1.2), resize_ratio=0.5, antialias=True):
        super().__init__()
        self.scale = scale
        self.resize_ratio = resize_ratio
        self.antialias = antialias

    def _get_params(self, flat_inputs):
        resize_scale = torch.rand(1) * (self.scale[1]-self.scale[0]) + self.scale[0]
        return dict(resize_scale=resize_scale)
    
    def _transform(self, inpt, params):
        resize = torch.rand(1) < self.resize_ratio
        if not resize:
            return inpt

        size = [int(params["resize_scale"]*inpt.shape[-2]), int(params["resize_scale"]*inpt.shape[-1])]
        
        return self._call_kernel(v2.functional.resize, inpt, size, antialias=self.antialias)


class Imgdataset(Dataset):
    def __init__(self, root_path='Dataset/DAVIS/JPEGImages/480p', crop_size=256, color=1, length=8, skip=1, cuda=False):

        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        else:
            assert isinstance(crop_size, list)
        
        self.device = "cuda" if cuda else "cpu"

        self.data = []
        self.length = length
        self.cvmode = 0 if color == 1 else 1
        self.pipeline = v2.Compose([
            MyRandomResize(resize_ratio=0.5, antialias=True),
            v2.RandomCrop(size=crop_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Resize(size=crop_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True)])

        root_path_list = sorted(os.listdir(root_path))
        index = 0
        self.data_map = {}
        for seti in root_path_list:
            seti_path = '{}/{}'.format(root_path, seti)
            data_path_list = sorted(os.listdir(seti_path))
            data_paths = []
            for datai in data_path_list:
                data_pathi = '{}/{}'.format(seti_path, datai)
                data_paths.append(data_pathi)

                if len(data_paths) == length:
                    self.data_map[index] = data_paths
                    index += 1
                    data_paths = data_paths[skip:]


    def __getitem__(self, index):
        imgs = []
        seq = self.data_map[index]
        for seq_i in seq:
            img = cv2.imread(seq_i, self.cvmode)
            imgs.append(img)

        imgs = torch.from_numpy(np.asarray(imgs)).to(self.device)
        if self.cvmode == 0:
            imgs = imgs.unsqueeze(1)
        else:
            imgs = torch.flip(imgs, dims=[3]).permute(0, 3, 1, 2)

        imgs = self.pipeline(imgs)
        # imgs [batch, time, color, h, w]
        return imgs

    def __len__(self):
        return len(self.data_map)
    

if __name__ == '__main__':
    from tqdm import tqdm
    train_dataloader = DataLoader(dataset=Imgdataset('Dataset/DAVIS/JPEGImages/480p', 256, 1, 8, cuda=False),  batch_size=8, num_workers=8, pin_memory=True)
    with tqdm(total=len(train_dataloader), ncols=150) as _tqdm:
        for iteration, batch in enumerate(train_dataloader):
            if iteration == 0 or iteration == len(train_dataloader)-1:
                print(batch.shape)
            _tqdm.update(1)