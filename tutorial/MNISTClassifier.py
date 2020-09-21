import os
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class Config():
    def __init__(self, img_w=28, img_h=28, hidden_size=128, output_size=10, batch_size=128, lr=1e-3, train_ratio=0.9,
                 epoch_size=10):
        self.img_w = img_w
        self.img_h = img_h
        self.input_size = img_w * img_h
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.lr = lr
        self.train_ratio = train_ratio
        self.epoch_size = epoch_size


class NN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.layer_1 = nn.Linear(config.input_size, config.hidden_size)
        self.layer_2 = nn.Linear(config.hidden_size, config.output_size)

        self.relu = nn.ReLU()
        self.cross_entropy = F.cross_entropy

    def forward(self, x):
        x = x.view(x.size(0), -1)   # (batch_size, input_size)
        x = self.relu(self.layer_1(x))
        predictions = self.layer_2(x)

        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy(y_hat, y)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # split dataset
        if stage == 'fit':
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            train_nums = int(len(mnist_train) * self.config.train_ratio)
            valid_nums = len(mnist_train) - train_nums
            self.mnist_train, self.mnist_val = random_split(mnist_train, [train_nums, valid_nums])
        if stage == 'test':
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.config.batch_size)


if __name__ == '__main__':
    config = Config()
    model = NN(config)
    data = MNISTDataModule(config)
    trainer = pl.Trainer(gpus=2, max_epochs=config.epoch_size)
    trainer.fit(model, data)
