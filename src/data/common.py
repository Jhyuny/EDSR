import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    #lr,hr image의 같은 부분을 비교하고 싶으므로, randomcrop()을 사용 불가
    #(randomcrop()은 random하게 뽑으므로 같은 부분을 비교할 수 없음)
    ih, iw = args[0].shape[:2] #args????

    if not input_large: #True 실행
        p = scale if multi else 1 #p = 1
        tp = p * patch_size #96
        ip = tp // scale #48
    else: 
        tp = patch_size # =96
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1) #random하게 설정
    iy = random.randrange(0, ih - ip + 1) 

    if not input_large: #True 실행
        tx, ty = scale * ix, scale * iy
    else: 
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip ],
        *[a[ty:ty + tp, tx:tx + tp ] for a in args[1:]]
    ]

    return ret 

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2: #흑백, 2차원일 때
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) #pytorch에서의 형식이 다르므로 수정
        tensor = torch.from_numpy(np_transpose.copy()).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True): #pytorch library사용 안하고, 마찬가지로 같은 augmentation을 취하기 위해
    hflip = hflip and random.random() < 0.5 #horizontal, randomly (p = 0.5)
    vflip = rot and random.random() < 0.5 #vertical, randomly (p = 0.5)
    #rot90 = rot and random.random() < 0.5 #rotation, randomly (p = 0.5)

    def _augment(img): #모두 True이므로 실행
        if hflip: img = img[:, ::-1 ]
        if vflip: img = img[::-1, : ]
        #if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

