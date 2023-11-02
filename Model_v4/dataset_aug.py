import torch
import cv2
import os
import numpy as np
from random import choices
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
    def __init__(self, root_path='Dataset/DAVIS/JPEGImages/480p', crop_size=256, color=1, length=8, skip=1):
        super(Imgdataset, self).__init__()
        self.data = []
        self.length = length
        self.transforms = v2.Compose([
                MyRandomResize(resize_ratio=0.5, antialias=True),
                v2.RandomCrop(crop_size),
                v2.ToDtype(torch.float32, scale=True)])
        self.cvmode = 0 if color == 1 else 1

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

        imgs = torch.from_numpy(np.asarray(imgs))
        if self.cvmode == 0:
            imgs = imgs.unsqueeze(1)
        else:
            imgs = torch.flip(imgs, dims=[3]).permute(0, 3, 1, 2)

        imgs = self.transforms(imgs)
        # imgs [length, C, crop_size, crop_size]
        return imgs

    def __len__(self):
        return len(self.data_map)

def transform(sample):
    # define transformations
    do_nothing = lambda x: x
    do_nothing.__name__ = 'do_nothing'
    flipud = lambda x: torch.flip(x, dims=[2])
    flipud.__name__ = 'flipup'
    rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
    rot90.__name__ = 'rot90'
    rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
    rot90_flipud.__name__ = 'rot90_flipud'
    rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
    rot180.__name__ = 'rot180'
    rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
    rot180_flipud.__name__ = 'rot180_flipud'
    rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
    rot270.__name__ = 'rot270'
    rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
    rot270_flipud.__name__ = 'rot270_flipud'
    add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), std=(5 / 255.)).expand_as(x).to(x.device)
    add_csnt.__name__ = 'add_csnt'

    # define transformations and their frequency, then pick one.
    aug_list = [do_nothing, flipud, rot90, rot90_flipud, rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
    w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12]  # one fourth chances to do_nothing
    transf = choices(aug_list, w_aug)

    # transform all images in array
    return transf[0](sample)

def normalize_augment(data, cuda=True, cr=24, start=0):
    # [bs, D, C, H, W]
    if cuda: data = data.cuda()
    data = data[:,start:start+cr,:,:,:]
    data_shape = data.size()
    data = data.view(data_shape[0], -1, data_shape[3], data_shape[4])
    data = transform(data).view(data_shape[0], data_shape[1], data_shape[2], data_shape[3], data_shape[4])
    return data


if __name__ == '__main__':
    from tqdm import tqdm
    train_dataloader = DataLoader(dataset=Imgdataset(),  batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    with tqdm(total=len(train_dataloader), ncols=150) as _tqdm:
        for iteration, batch in enumerate(train_dataloader):
            batch = normalize_augment(batch, cr=8)
            _tqdm.update(1)