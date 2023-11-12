import time
from collections import OrderedDict
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import pytorch_lightning as pl

class EvaluationModelHalfKP(pl.LightningModule):
  def __init__(self,learning_rate=1e-3,batch_size=1024, arch=[776,500,500,1]):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate

    self.shared_linear = nn.Linear(45080, 256)
    self.concat_linear = nn.Linear(512, 13)
    self.output_linear = nn.Linear(13, 1)

  def forward(self, input1, input2):
    # Shared linear layer
    x1 = F.relu(self.shared_linear(input1.float()))
    x2 = F.relu(self.shared_linear(input2.float()))

    # Concatenate and further processing
    x = torch.cat((x1, x2), dim=1)
    x = F.relu(self.concat_linear(x))
    x = self.output_linear(x)
    return x

  def training_step(self, batch, batch_idx):
    input1, input2, y = batch['ego_bin'], batch['alter_bin'], batch['eval']
    y_hat = self(input1, input2)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class EvaluationModel(pl.LightningModule):
  def __init__(self, learning_rate=1e-3,batch_size=1024,arch=[776,500,500,1]):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    for i in range(1, len(arch)-1):
      layers.append((f"linear-{i}", nn.Linear(arch[i-1], arch[i])))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{len(arch)-1}", nn.Linear(arch[-2], arch[-1])))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
  
class EvaluationModelDual(pl.LightningModule):
  def __init__(self, learning_rate=1e-3,batch_size=1024,arch=[776,1000,1000,1]):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate

    self.w_ft = nn.Linear(arch[0], arch[1])
    self.b_ft = nn.Linear(arch[0], arch[1])

    layers = []
    for i in range(1, len(arch)-1):
      layers.append((f"linear-{i}", nn.Linear(arch[i-1], arch[i])))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{len(arch)-1}", nn.Linear(arch[-2], arch[-1])))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x_w, x_b):
    # TODO (vrubies)
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x_w, x_b, y = batch['binary_white'], batch['binary_black'], batch['eval']
    y_hat = self(x_w, x_b)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)