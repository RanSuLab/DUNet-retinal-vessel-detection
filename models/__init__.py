import numpy as np

np.random.seed(0)

from models.deform import *
from models.deform_unet import *
from models.unet import UNet

MODELS = {'unet': UNet,
          'deform_v1': DeformConvNetV1V2,
          'deform_unet_v1': DUNetV1V2,
          }
