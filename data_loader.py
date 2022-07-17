"""
Modified by Yujin Oh
https://github.com/yjoh12/CXR-Segmentation-by-AdaIN-based-Domain-Adaptation-and-Knowledge-Distillation.git

Forked from StarGAN v2, Copyright (c) 2020-preeent NAVER Corp.
https://github.com/clovaai/stargan-v2.git
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image, ImageFilter
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
import glob


DOMAIN_MASK = 2
DOMAIN_PAIRED = 1
DOMAIN_UNPAIRED = 0


def listdir(dname):
    tag_list = ['.png', '.jpg'] 
    fnames = list(chain(*[list(Path(dname).rglob('*' + ext)) for ext in tag_list]))
    return fnames


class SourceMaskTrainDataset(data.Dataset):
    def __init__(self, root, transform=None, flag_seg=False, args=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.args = args
        self.Unsharp = ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3)
        self.Blur = ImageFilter.BoxBlur(radius=2)

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, labels = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            cls_fnames.sort()
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        self.mask_idx = list(range(labels.index(idx), len(labels)))
        return fnames, labels

    def __getitem__(self, index):

        fname = self.samples[index]
        fname_str = str(fname)
        idx_msk = np.random.choice(self.mask_idx)
        fname_msk = str(self.samples[idx_msk])
        target_msk = DOMAIN_MASK
        flag_pair = False
        if fname_str.find(self.args.domain1)>=0:
            frame_msk_cand = glob.glob(fname_str.replace('image.', 'mask.').replace(self.args.domain1, self.args.domain2))
            if len(frame_msk_cand)>0:
                fname_msk = frame_msk_cand[0]
                idx_msk = index
                flag_pair = True
        
        # data loader
        fname_str = str(fname)
        flag_img = True
        if (fname_str.find('_gt')>=0) | (fname_str.find('_mask')>=0) | (fname_str.find('_roi')>=0) :
            flag_img=False
        img = (np.array(Image.open(fname_str).convert('L'))/255.0).astype(np.float32)
        msk = (np.array(Image.open(fname_msk).convert('L'))/255.0).astype(np.float32)
        flag_train = True      
        img = aug_cxr(img, self.args, flag_pair = flag_pair, flag_img=flag_img, flag_train=flag_train)
        msk = aug_cxr(msk, self.args, flag_pair = flag_pair, flag_img=flag_img, flag_train=flag_train)

        return img, self.targets[index], msk, target_msk 

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None, args=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.args = args

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        fname_str, fname2_str = str(fname), str(fname2)
        flag_img=True
        if (fname_str.find('_gt')>=0) | (fname_str.find('_mask')>=0) | (fname_str.find('_roi')>=0):
            flag_img=False
        
        # data loader
        img = (np.array(Image.open(fname_str).convert('L'))/255.0).astype(np.float32)
        img2 = (np.array(Image.open(fname2_str).convert('L'))/255.0).astype(np.float32)
        img = aug_cxr(img, self.args, flag_pair = False, flag_img=flag_img, flag_train=True)
        img2 = aug_cxr(img2, self.args, flag_pair = False, flag_img=flag_img, flag_train=True)
        label = self.targets[index]

        return img, img2, label

    def __len__(self):
        return len(self.targets)


class DefaultEvalDataset(data.Dataset):
    def __init__(self, root, transform=None, ext_flag=False, test_flag=False, args=None):
        self.ext_flag = ext_flag
        self.test_flag = test_flag
        self.samples = self._make_dataset(root)
        self.transform = transform
        self.args = args

    def _make_dataset(self, root):
        fnames = listdir(root)
        fnames.sort()
        return fnames

    def __getitem__(self, index):

        fname = self.samples[index]
        fname_str = str(fname)
        flag_img=True
        if (fname_str.find('_gt')>=0) | (fname_str.find('_mask')>=0) | (fname_str.find('_roi')>=0):
            flag_img=False
            
        # data loader
        img = (np.array(Image.open(fname_str).convert('L'))/255.0).astype(np.float32)
        img = aug_cxr(img, self.args, flag_pair = False, flag_img=flag_img)

        return img

    def __len__(self):
        return len(self.samples)


def aug_cxr(img, args, flag_pair = False, flag_img=False, flag_train=False):
    
    size = (args.img_size, args.img_size) 

    if flag_img:
        img = (img-img.min())/(img.max()-img.min())*255

    # augmentation
    img = [Image.fromarray(np.array(img).astype(np.uint8))]   

    # resize
    if flag_img:
        img = [Image.fromarray(np.array(TF.resize(Image.fromarray(np.array(slices).clip(0,255).astype(np.uint8)), size, interpolation=transforms.InterpolationMode.BILINEAR))) for slices in img]
    else:
        img = [Image.fromarray(np.array(TF.resize(Image.fromarray(np.array(slices).astype(np.uint8)), size, interpolation=transforms.InterpolationMode.BILINEAR))) for slices in img]
    
    # to torch
    img = torch.tensor(np.array(img[0])).unsqueeze(0)

    # stack 
    if flag_img:
        img = img/255.0

    # normalize
    img = img*2.0-1.0

    return img


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, num_workers=4, args=None):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([img_size, img_size], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), 
    ])

    if which == 'source_mask':
        dataset = SourceMaskTrainDataset(root, transform, args=args)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform, args=args)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32, shuffle=False, num_workers=4, drop_last=False, ext_flag=False, test_flag=False, args=None):

    height, width = img_size, img_size
    mean = [0.5]
    std = [0.5] 
        
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([height, width], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultEvalDataset(root, transform=transform, ext_flag=ext_flag, test_flag=test_flag, args=args)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_inputs_w_mask(self):
        try:
            x, y, x_msk, y_msk = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, x_msk, y_msk = next(self.iter)
        return x, y, x_msk, y_msk

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        if self.mode == 'train_mask':
            x, y, x_msk, y_msk = self._fetch_inputs_w_mask()
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            z_self_cons = torch.ones(x.size(1), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2, z_self=z_self_cons,
                           x_msk=x_msk, y_msk=y_msk)
        elif self.mode == 'val':
            x, y = self._fetch_inputs()
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            x, y = self._fetch_inputs()
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
