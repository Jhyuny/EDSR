import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale #args.scale = 4
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data) #args.dir_data = dataset이 있는 경로
        if args.ext.find('img') < 0: #args.ext = 'sep' 이므로 if 조건문은 True: 실행
            path_bin = os.path.join(self.apath, 'bin') #self.apath는 밑의 _set_filesystem에서 정의.
            #self.apath = dir_data이므로 path_bin = dir_data/bin
            os.makedirs(path_bin, exist_ok=True) #directory를 만든다.

        list_hr, list_lr = self._scan() #scan의 return값이 각각 list_hr, list_lr이 된다.
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0: #여기에 해당 args.ext = 'sep'이므로
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True #해당 directory가 이미 있을 경우 에러 발생 없이 넘어가고, 없을 경우에만 생성한다.
            )
            for s in self.scale: #self.scale = '4' ????
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True #해당 directory가 이미 있을 경우 에러 발생 없이 넘어가고, 없을 경우에만 생성한다.
                )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]#scale에 따라서 iteration, x2,x3,x4의 sacle을 한 번에 path를 지정해주기 위해 lr은 list 원소를 list로 사용
            for h in list_hr:
                b = h.replace(self.apath, path_bin) #경로를 path_bin (bin폴더가 있는 path로 바꿔준다)
                b = b.replace(self.ext[0], '.pt') #.png > .pt로 변경 / self.ext[0] = '.png' 
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) #_check_and_load (파일입력)
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
        if train:#True이므로 실행
            n_patches = args.batch_size * args.test_every #batch_size = 16, test_every = 1000 / train 수를 늘려서 
            n_images = len(args.data_train) * len(self.images_hr) #????? args.data_train이 뭐지
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1) #n_patches를 n_images로 나눈 몫 만큼 반복

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted( #낮은 순서대로 정렬
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])) #self.ext = ('.png', '.png') 로 _set_filesystem에 정의
        )#self.dir_hr/*.png / glob()이므로 '.png'를 포함하는 파일을 search
        names_lr = [[] for _ in self.scale] #names_lr = [[]]..?
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )#names_lr의 si번째 index에 dir_data/LR_bicubic/X(scale)/(filename)x(scale)(.png)로 대입
                ))

        return names_hr, names_lr #lr,hr image 각각의 path를 list에 정리

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)  #self.apath = dir_data + '' = dir_data
        self.dir_hr = os.path.join(self.apath, 'HR') #dir_data/HR
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic') #dir_data/LR_bicubic
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True): #self._check_and_load(args.ext, h, b, verbose=True)
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose: #실행
                print('Making a binary: {}'.format(f)) #f = b, b는 .pt파일 경로
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f) #pickle .txt이외의 자료형을 파일로 저장할 때 사용하는 모듈
                #picke.dump(data,file) 형식으로 사용

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx) # line 120, memory에 올라온 직접적인 영상, image가 저장
        pair = self.get_patch(lr, hr) #get_patch : 
        pair = common.set_channel(*pair, n_channels=self.args.n_colors) #n_channels=self.args.n_colors = 3
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range) #rgb_range=self.args.rgb_range = 255

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train: #True실행
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx): 
        if self.train: #True 실행
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx) #line114 / _get_index를 이용해서 불러온다.
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr)) #imageio나 pickle을 통해 불러온 image를 memory에 입력
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr) #image불러올 땐 imageio
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f) #'sep'은 pickle로 저장.
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr): #이미지 일부 추출 / image크기 통일시키기 위해 result 형식 : [batchsize,h,w,c]
        #resize해서 통일하기에는 무리가 있음 > 원본 image를 그대로 사용해야하므로 get_patch를 이용해서 일부 사용.
        #모든 계산 연산값을 저장해야하므로 200*200만 되도 ram터져..
        scale = self.scale[self.idx_scale] #self.idx_scale = 0 
        if self.train: #실행
            lr, hr = common.get_patch( #<common.py>, lr = ret[0], hr = ret[1]
                lr, hr,
                patch_size=self.args.patch_size, #192 
                scale=scale, #2
                multi=(len(self.scale) > 1), #True
                input_large=self.input_large #False
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr) #no_augment = False / 실행 
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

