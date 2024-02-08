import segmentation_models_pytorch as smp
import torch

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)



class Criterion(object):
    def __init__(self) :
        pass

    def __call__ (self, y_pred, y_true):
        return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

#######################
#                     #
#######################

def bce_lovasz(output, target):
    return (0.5 * BCELoss(output, target)) + (0.5 * LovaszLoss(output, target))

def bce_lovasz_tversky_loss(output, target):
    return (0.25 * BCELoss(output, target)) + (0.25 * LovaszLoss(output, target)) + (0.5 * TverskyLoss(output, target))

def get_loss(epoch):
    if epoch <= 5:
        return bce_lovasz
    else:
        return bce_lovasz_tversky_loss