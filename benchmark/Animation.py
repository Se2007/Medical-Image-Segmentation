import torch 
import cv2
import pandas as pd 
import numpy as np
from PIL import Image
import imageio
import glob
from torchvision import transforms as T
from torchvision.transforms.functional import to_tensor

# root = './UW_madison_dataset/train/case22/case22_day0/scans/slice_*.png'
root = './UW_madison_dataset/train/case33/case33_day0/scans/slice_*.png'


list_path = glob.glob(root)
image_lst = []


def rle2mask(image_size, segments):
    mask = torch.zeros(3, image_size[0] * image_size[1], dtype=torch.float32)
    for idx, segment in enumerate(segments):
        if str(segment) != 'nan':
            segment = segment.split(' ')
            starts = np.array(segment[::2], dtype=np.int32)  - 1
            lengths = np.array(segment[1::2], dtype=np.int32) 
            ends = starts + lengths
            for s, e in zip(starts, ends):
                mask[idx, s:e] = 1

    return mask.reshape((3, image_size[0], image_size[1]))

image_trasform = T.Compose([T.Resize((512, 512)),
                            T.ToTensor(),
                            T.Lambda(lambda x : (x - x.min()) / (x.max() - x.min())),
                            T.Lambda(lambda x : x.repeat(3, 1, 1))])

mask_trasform = T.Compose([T.Resize((512, 512))])

train_df = pd.read_csv('./UW_madison_dataset/' + 'mask.csv')

for path in list_path :
    id = path.split('/')[-2]+ '_slice_'+path.split('/')[-1].split('_')[-5]

    sample = train_df[train_df['id'] == id]
    # print(sample['height'].values, sample['weidth'], len(sample))
    if len(sample) > 0 :
        image = image_trasform(Image.open(path))
        mask = mask_trasform(rle2mask([sample.iloc[0]['height'], sample.iloc[0]['weidth']], [sample.iloc[0]['large_bowel'], sample.iloc[0]['small_bowel'], sample.iloc[0]['stomach']]))

        out = image.permute(1,2,0).numpy() + mask.permute(1,2,0).numpy() * 0.55

    else : 
        out = image_trasform(Image.open(path)).permute(1,2,0).numpy()
       
    frame_rgb = (out * 255).astype(np.uint8)
    image_lst.append(frame_rgb)

    cv2.imshow('Red : large_bowel -- Green : small_bowel -- Blue : stomach', out)
    cv2.waitKey(100)

imageio.mimsave('./v.gif', image_lst)