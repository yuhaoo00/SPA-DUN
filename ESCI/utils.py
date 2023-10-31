import argparse
import random
import torch
import numpy as np
from PIL import Image

def save_img(img, path):
    if img.shape[-1] == 1 or len(img.shape) == 2:
        img = np.squeeze(img)
        mode = 'L'
    else:
        mode = 'RGB'
    img = Image.fromarray(img.astype('uint8'), mode=mode)
    img = img.save(path)
    return

def save_gif(imgs, path):
    if imgs.shape[-1] == 1 or len(imgs.shape) == 3:
        imgs = np.squeeze(imgs)
        mode = 'L'
    else:
        mode = 'RGB'
    imgs_list = []
    for i in range(imgs.shape[0]):
        img = Image.fromarray(imgs[i,...].astype('uint8'), mode=mode).convert('RGB')
        imgs_list.append(img)
    imgs_list[0].save(path, save_all=True, append_images=imgs_list, loop=0, duration=0.2)
    return

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def get_sd(path, cuda):
    model = torch.load(path, map_location='cpu') if not cuda else torch.load(path, map_location='cuda')
    if 'model_state_dict' in model.keys():
        model = model['model_state_dict']
    elif 'state_dict' in model.keys():
        model = model['state_dict']
    elif 'color_SCI_backward_dict' in model.keys():
        model = model['color_SCI_backward_dict']
    return model

def load_model(model, path, cuda=True):
    pretrain_sd = get_sd(path, cuda)
    model_sd = model.state_dict()
    pretrain_sd = {k: v for k, v in pretrain_sd.items() if k in model_sd}
    model.load_state_dict(pretrain_sd, strict=False)

    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dict2namespace(config, namespace=None):
    if namespace is None:
        namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def print_args(namespace, logger=None):
    if logger is not None:
        logger.info('[Command Args]')
    else:
        print('[Command Args]')

    for key, value in namespace.__dict__.items():
        txt = "--{}: {}".format(key, value)
        if logger is not None:
            logger.info(txt)
        else:
            print(txt)