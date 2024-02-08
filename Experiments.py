import torch
import os
import time

from torchmetrics import Dice
from torchmetrics.aggregation import MeanMetric
import segmentation_models_pytorch as smp

from benchmark import dataset
from methods import unet, unetplusplus

from prettytable import PrettyTable
from colorama import Fore, Style, init


## Arguments
device = 'cpu'
load_path = './saved_model/unet-efficientnet-encoder.pth'


def segment(image, model):
  with torch.inference_mode():
    prediction = model(image)
    return torch.sigmoid(prediction)


##  Load test data_loader
test_batch_size = 64
test_loader = dataset.UW_madison(root='./benchmark/UW_madison_dataset', mode='test', mini=False, memory=False)(batch_size=test_batch_size)

## Load Model

# model = unet.UNet(n_channels=3, n_classes=3, bilinear=False).to(device)
model = unet.pre_train_unet().to(device)
# model = unetplusplus.UnetPlusPlus(encoder_name='efficientnet-b3').to(device)
# model = unetplusplus.UnetPlusPlus(encoder_name='resnet18').to(device)


sate = torch.load(load_path)
model.load_state_dict(sate['state_dict'])



def evaluate(model, test_loader, device='cpu'):
  model.eval().to(device)
  iou_score, f1_score, f2_score, accuracy, recall = MeanMetric(), MeanMetric(), MeanMetric(), MeanMetric(), MeanMetric()
  dice_metric = Dice(average='micro').to(device)

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)

    #   outputs = model(inputs)
      outputs = segment(inputs, model)
    
      targets = targets.to(torch.int32)##.update(loss.item(), weight=len(targets))

      tp, fp, fn, tn = smp.metrics.get_stats(outputs.cpu(), targets.cpu(), mode='multilabel', threshold=0.5)
      
      iou_score.update(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"), weight=len(targets))
      f1_score.update(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"), weight=len(targets))
      f2_score.update(smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro"), weight=len(targets))
      accuracy.update(smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro"), weight=len(targets))
      recall.update(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise"), weight=len(targets))

      dice_metric(outputs, targets)

  return iou_score.compute().item(), f1_score.compute().item(), f2_score.compute().item(), accuracy.compute().item(), recall.compute().item(), dice_metric.compute().item()



iou_score, f1_score, f2_score, accuracy, recall, dice = evaluate(model, test_loader, device='cuda')

table = PrettyTable()

# Define column names and alignment
table.field_names = ["Metric", "Value"]
table.align["Metric"] = "l"
table.align["Value"] = "r"

table.add_row(["IoU Score", f"{iou_score:.2%}"])
table.add_row(["F1 Score", f"{f1_score:.2%}"])
table.add_row(["F2 Score", f"{f2_score:.2%}"])
# table.add_row(["Accuracy", f"{accuracy:.2%}"])
table.add_row(["Recall", f"{recall:.2%}"])
table.add_row(["Dice", f"{dice:.2%}"])

print(table)


