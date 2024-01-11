import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision import tv_tensors

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

import cv2

def rle2mask(image_size, segments):
    mask = torch.zeros(3, image_size[0] * image_size[1], dtype=torch.int32)
    for idx, segment in enumerate(segments):
        if str(segment) != 'nan':
            segment = segment.split(' ')
            starts = np.array(segment[::2], dtype=np.int32)  - 1
            lengths = np.array(segment[1::2], dtype=np.int32) 
            ends = starts + lengths
            for s, e in zip(starts, ends):
                mask[idx, s:e] = 1

    return mask.reshape((3, image_size[0], image_size[1]))

class UW_madison_dataset(Dataset):
    def __init__(self, csv_file_path, transforms, memory=False):
        self.csv_file = pd.read_csv(csv_file_path)
        self.transforms = transforms

        self.memory = memory
        if memory:
            self._save_memory()

    def __len__(self,):
        return len(self.csv_file)
    
    def __getitem__(self, index):
         
        sample = self.csv_file.iloc[index]

        image = self.imgs[index] if self.memory else self._load_image(sample['Address'])
        mask = tv_tensors.Mask(rle2mask([sample['height'], sample['weidth']], [sample['large_bowel'], sample['small_bowel'], sample['stomach']]))

        image, mask = self.transforms(image, mask)

        return image, mask 
    
    def _save_memory(self):
        self.imgs = []
        for path in self.csv_file['Address']:
            self.imgs.append(self._load_image(path))

    def _load_image(self, path):
        img = cv2.imread('./benchmark'+path[1:], cv2.IMREAD_UNCHANGED)  ## './benchmark'+path[1:] -->>> It works for now After I have to make new csv file that don't have this bug
        img = self._minmax_scaler(img)
        return tv_tensors.Image(img)

    def _minmax_scaler(self, x):
        return np.array((x - x.min()) / (x.max() - x.min()), dtype=np.float32)
    
class UW_madison(object):
    def __init__(self, root, mode, mini=False, memory=True) :
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini  
        self.memory = memory
        
        self.transform = v2.Compose([v2.Resize(size=(234, ), antialias=True),
                           v2.RandomCrop(size=(224, 224)),
                           v2.RandomPhotometricDistort(p=.5),
                           v2.RandomHorizontalFlip(p=.5),
                           v2.ElasticTransform(alpha=50),
                           v2.ToTensor(),
                           v2.Lambda(lambda x : (x - x.min()) / (x.max() - x.min())),
                           v2.Normalize(mean=(0.5, ), std=(0.5, )),
                           v2.Lambda(lambda x : x.repeat(3, 1, 1))
                           ])
            

        if mode == 'train' :
            self.path_dataset = root + "/train.csv"
        elif mode == 'valid':
            self.path_dataset = root + "/valid.csv"
        elif mode == 'test' :
            self.path_dataset = root + "/test.csv"


    def __call__(self, batch_size) :
        dataset = UW_madison_dataset(self.path_dataset, self.transform, memory=self.memory)

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

        elif self.mini == True:
            dataset,_ = random_split(dataset,(1000, len(dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader



if __name__=='__main__':
    '''
    ## show the image and mask with custom dataset

    trasform = v2.Compose([v2.Resize(size=(234, ), antialias=True),
                           v2.RandomCrop(size=(224, 224)),
                           v2.RandomPhotometricDistort(p=.5),
                           v2.RandomHorizontalFlip(p=.5),
                           v2.ElasticTransform(alpha=50),
                           v2.ToTensor(),
                           v2.Lambda(lambda x : (x - x.min()) / (x.max() - x.min())),
                           v2.Lambda(lambda x : x.repeat(3, 1, 1))
                           ])

    path = "./UW_madison_dataset/mask.csv"
    dataset = UW_madison_dataset(path, trasform, memory=True)

    index = random.randint(0, dataset.__len__()) 
    mask, image = dataset.__getitem__(index)

    plt.imshow(image.permute(1,2,0) + mask.permute(1,2,0) * 0.5)
    plt.show()
    '''
    ## show the image and mask with dataloader

    dataloader = UW_madison(root='./UW_madison_dataset', mode='train', mini=True, memory=True)(batch_size=32)

    mask, img = next(iter(dataloader))

    print(img[0].min(), img[0].max())
    plt.imshow(img[0].permute(1,2,0) + mask[0].permute(1,2,0) * 0.5)
    plt.show()


     