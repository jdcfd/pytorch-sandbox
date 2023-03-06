import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from model.param_net import ParamNet
from model.shape_net import ShapeNet

class NIF_small(pl.LightningModule):
    def __init__(self,act='silu',learning_rate=0.001):
        super().__init__()
        if act == 'silu':
            self.act_fn = F.silu
            self.param_net = ParamNet(act_fn=nn.SiLU)
        if act == 'tanh':
            self.act_fn = F.tanh
            self.param_net = ParamNet(act_fn=nn.Tanh)
        if act == 'relu':
            self.act_fn = F.relu
            self.param_net = ParamNet(act_fn=nn.ReLU)
        self.shape_net = ShapeNet
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, t, u = batch
        params = self.param_net(t)
        pred = self.shape_net(x,params,self.act_fn)
        loss = F.mse_loss(pred, u)
        self.log("train_loss", loss)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, t, u = batch
        params = self.param_net(t)
        pred = self.shape_net(x,params,self.act_fn)
        loss = F.mse_loss(pred, u)
        self.log("test_loss", loss)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

