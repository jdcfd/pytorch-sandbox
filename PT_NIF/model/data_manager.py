import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import numpy as np

class DataManager(pl.LightningDataModule):
    def __init__(self,data_file,batch_size=32):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.transform = ToTensor()

    def setup(self, stage=None):
        x,t,u = torch.split(torch.Tensor(np.load(self.data_file)['data']),1,1)
        data = TensorDataset(x,t,u)
        self.train_data, self.test_data = random_split(data, [0.8,0.2],
            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
