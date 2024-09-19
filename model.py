import torch
import torch.nn as nn
import numpy as np
import argparse
import load_data
from Plot import plot_spectrum
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=5, help="number of classes for dataset")
parser.add_argument("--sp_size", type=int, default=1000, help="size of each spectra length")
parser.add_argument("--par_size", type=int, default=5, help="size of each parameters length")
parser.add_argument("--epoch_interval", type=int, default=10, help="iteration times")
parser.add_argument("--sample_interval", type=int, default=10, help="sample_interval for test")
opt =parser.parse_args()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(opt.par_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,1000),
            nn.ReLU(),
            nn.Linear(1000,3000),
            nn.Tanh()
            )
        self.conv = nn.Conv1d(1,3,kernel_size=3)
    def forward(self,labels):
        labels = labels.double()
        x=self.fc1(labels)
        return x
