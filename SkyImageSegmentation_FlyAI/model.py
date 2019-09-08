# -*- coding: utf-8 -*
import os
import torch
from flyai.model.base import Base
from torch.autograd import Variable
from path import MODEL_PATH
from fcn_segmentation import FCN16s

Torch_MODEL_NAME = "model.pkl"

cuda_avail = torch.cuda.is_available()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        cnn = FCN16s(1).to(device)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        cnn.to(device)
        cnn.eval()
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        x_data = x_data.float().to(device)
        outputs = cnn(x_data)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu()
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        print(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        cnn = FCN16s(1).to(device)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        cnn.to(device)
        cnn.eval()
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            x_data = torch.from_numpy(x_data)
            x_data = x_data.float().to(device)
            outputs = cnn(x_data)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu()
            prediction = outputs.data.numpy()
            prediction = self.data.to_categorys(prediction)
            labels.append(prediction)
        return labels


    def save_model(self, network, path, name=Torch_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network.state_dict(), os.path.join(path, name))