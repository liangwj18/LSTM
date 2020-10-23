import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import json
import csv
import random
from PIL import Image

import os
import os.path

import cv2
import glob

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (S x T x C x H x W)
    to a torch.FloatTensor of shape (S x C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    #print(pic.shape)
    return pic.transpose([0,2,1,3,4])

def shot_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x C x H x W)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    #print(pic.shape)
    return pic.transpose([1,0,2,3])



# TODO
# def load_flow_frames(image_dir, vid, start, num):
#   frames = []
#   for i in range(num):
#     imgx = cv2.imread(os.path.join(image_dir, str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
#     imgy = cv2.imread(os.path.join(image_dir, str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
#     w,h = imgx.shape
#     if w < 224 or h < 224:
#         d = 224.-min(w,h)
#         sc = 1+d/min(w,h)
#         imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
#         imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
#     imgx = (imgx/255.)*2 - 1
#     imgy = (imgy/255.)*2 - 1
#     img = np.asarray([imgx, imgy]).transpose([1,2,0])
#     frames.append(img)
#   return np.asarray(frames, dtype=np.float32)

def make_dataset(root, mode, std_shot):
    data_path = []
    dataset = []
    # print("root",root)

    videos = sorted(glob.glob(root + "/*"))
    # print(videos)
    for video in videos:
        shots = sorted(glob.glob(video + "/*"))
        # print(video, len(shots))
        if len(shots)<=std_shot:
            continue

        data_path.append([])
        for shot in shots:
            path = os.path.join(shot, mode)
            nf = len(glob.glob(path + "/*.*"))
            data_path[len(data_path)-1].append((path, nf))

        for i in range(len(shots)-std_shot):
            dataset.append((len(data_path)-1, i))

    return data_path, dataset


class MontageDataset(data_utl.Dataset):

    def __init__(self, root, train_mode, mode, transforms=None, std_frame=10, std_shot=5):
        
        self.root = os.path.join(root, train_mode)
        self.train_mode = train_mode
        self.mode = mode
        self.transforms = transforms
        self.std_frame = std_frame
        self.std_shot = std_shot
        
        self.data_path, self.data = make_dataset(self.root, self.mode, self.std_shot)

    def load_rgb_frames(self, image_dir, num):
        frames = []
        for i in range(1, num+1):
            img = Image.open(os.path.join(image_dir, str(i)+'.jpg'))
            img = self.transforms(img)
            frames.append(img)
            #print(i)
        return torch.stack(frames, 0)
        
    def load_shot(self, vid, shot):
        path, nf = self.data_path[vid][shot]
        if nf>self.std_frame:
            nf = self.std_frame
        if self.mode == 'rgb':
            imgs = self.load_rgb_frames(path, nf)
        # TODO
        # else:
        #     imgs = load_flow_frames(path, nf)
        if (imgs.shape[0]<self.std_frame):
            imgs = F.pad(imgs, (0,0,0,0,0,0,0, self.std_frame-imgs.shape[0]),"constant", 0)
        return imgs

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, shot = self.data[index]
        shots = []
        for i in range(self.std_shot-1):
            shots.append(self.load_shot(vid, shot+i))
        shots = torch.stack(shots, 0).permute(0,2,1,3,4)
        
        real_shot=self.load_shot(vid, shot+self.std_shot-1).permute(1,0,2,3)

        # if self.train_mode=="train":
        ns = random.randint(0,len(self.data_path[vid])-1)
        while (ns>=shot and ns<shot+self.std_shot):
            ns = random.randint(0,len(self.data_path[vid])-1)
        # print("shot",shot,"len",len(self.data_path[vid]),"ns",ns,"i",i)
        fake_shot=self.load_shot(vid, ns).permute(1,0,2,3)
        # else:
        #     fake_shot = []
        #     for i in range(shot):
        #         fake_shot.append(self.load_shot(vid, i))
        #     for i in range(shot+self.std_shot, len(self.data_path[vid])):
        #         fake_shot.append(self.load_shot(vid, i))
        #     fake_shot = torch.stack(fake_shot, 0).permute(0,2,1,3,4)
            
        return shots, real_shot, fake_shot

    def __len__(self):
        return len(self.data)
