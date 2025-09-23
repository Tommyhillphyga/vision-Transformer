import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm, trange
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from vit import Vit
np.random.seed(0)
torch.manual_seed(0)



def main (): 
    transform = ToTensor()
    train_set = MNIST(root='./../datasets', train=True, download = True, transform = transform)
    test_set = MNIST(root='./../datasets', train=False, download = True, transform = transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Vit()

    