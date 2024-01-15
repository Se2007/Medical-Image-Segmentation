import torch
import pandas as pd 
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from benchmark import dataset
import segmentation_models_pytorch as smp
from Unet import unet


norm = lambda x : (x - x.min()) / (x.max() - x.min())

def segment(image, model):
  with torch.inference_mode():
    prediction = model(image)
    return torch.sigmoid(prediction)

## Arguments
device = 'cpu'
load_path = './saved_model/unet-efficientnet-encoder.pth'  #

## Load Model

# model = unet.UNet(n_channels=3, n_classes=3, bilinear=False).to(device)
model = smp.Unet(encoder_name='efficientnet-b1', encoder_weights='imagenet',
                in_channels=3, classes=3).to(device)

sate = torch.load(load_path)
model.load_state_dict(sate['state_dict'])

##  Load test data_loader

test_batch_size = 1
test_loader = dataset.UW_madison(root='./benchmark/UW_madison_dataset', mode='test', mini=False, memory=False)(batch_size=test_batch_size)
image, mask = next(iter(test_loader))

## Get the segmentation from Model

output = segment(image, model)

## Show Result

print('\nRed ->> large_bowel -- Green ->> small_bowel -- Blue ->> stomach')

plt.title('Red : large_bowel -- Green : small_bowel -- Blue : stomach')
grid_image = make_grid([norm(image).squeeze(0) + norm(mask).squeeze(0) * 0.55, norm(image).squeeze(0), norm(image).squeeze(0) + output.squeeze(0) * 0.75], nrow=3)
plt.imshow(grid_image.permute(1,2,0))
plt.axis('off')
plt.show()