from keras.models import load_model
import torch
import torch.nn as nn
from collections.abc import Iterable

# from tensorflow.python.keras import backend as K

import onnx
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx

onnx_model_name = 'fish-resnet50.onnx'


model = load_model('model-resnet50-final.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)
import onnx
from onnx2pytorch import ConvertModel

onnx_model = onnx.load(onnx_model_name)
pytorch_model = ConvertModel(onnx_model)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = load_model('GN_W0.5_S2_ArcFace_epoch16.h5')

