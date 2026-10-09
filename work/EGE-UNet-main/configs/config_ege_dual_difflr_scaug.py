from configs.config_ege_dual_difflr import ege_dual_difflr_config
from torchvision import transforms
from utils import (myNormalize, myToTensor, myRandomHorizontalFlip,
                   myRandomVerticalFlip, myRandomRotation, myResize,
                   myRandomScaleCrop)
from datetime import datetime


class ege_dual_difflr_scaug_config(ege_dual_difflr_config):
    network = 'ege_dual'

    scale_aug_p = 0.5
    scale_aug_range = (0.5, 2.0)

    train_transformer = transforms.Compose([
        myNormalize('isic18', train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myRandomScaleCrop(p=scale_aug_p, scale_range=scale_aug_range),
        myResize(256, 256)
    ])

    work_dir = 'results/ege_dual_difflr_scaug_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
