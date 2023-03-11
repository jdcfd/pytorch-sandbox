# import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
# import argparse
from model.nif import NIF_small
from model.data_manager import DataManager

data_file = 'data/tw_train.npz'

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight,std=1e-1)
        m.bias.data.fill_(0.01)

data = DataManager(data_file,256)
model = NIF_small()
model.apply(init_weights)

trainer = pl.Trainer()
trainer.fit(model,data)