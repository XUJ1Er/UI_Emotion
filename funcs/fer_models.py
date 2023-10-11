"""
This file contains Facial Expression Recognition method.
"""

import cv2
import torch

from funcs.model import vgg16face
from funcs.model import FERModel

model_list = ["None", "vgg16", "MFER"]


def createFERmodel(model_name, weights_dir="random", cuda=True):
    model = None
    if model_name == "vgg16":
        model = vgg16face(weights_dir=weights_dir, cuda=cuda)

    elif model_name == "MFER":
        model = FERModel(weight_dir=weights_dir, cuda=cuda)

    else:
        raise Exception("Unknown FER model.")

    return model
