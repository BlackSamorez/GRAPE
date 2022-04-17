import torch
import torch.nn as nn
from torch.optim import Adam

import math

from tqdm.auto import tqdm

from .multiqubitgates import Pulse, CXCascade, DummyInput

class Constructor():
    def __init__(self, target):
        self.size = (target.shape[0]-1).bit_length()
        self.target = target.clone().detach()
        
        self.model = torch.nn.Sequential(*self.get_modules(self.size))
        for i in range(len(self.model)):
            self.model[i].randomize_params()
            
        self.optimizer = Adam(self.model.parameters())
        
    def get_modules(self, size):
        modules = [DummyInput(size)]
        
        for i in range(2 * size):  # TODO: find optimal
            modules.append(Pulse(size))
            modules.append(CXCascade(size))
        modules.append(Pulse(size))
        
        return modules
    
    def loss(self, mat):
        diff = mat - self.target
        return torch.trace(diff @ diff.T.conj())
    
    def train(self, it=100):
        pbar = tqdm(range(it), total=it)
        losses = []

        for i in pbar:
            mat = self.model(0)  # dummy input
            loss = self.loss(mat)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            pbar.set_description(
                f"Loss: {round(loss.item().real, 4)}"
            )

            losses.append(loss.item().real)
            
        return losses
            
    def to(self, device):
        self.model.to(device)
        self.target.to(device)
        