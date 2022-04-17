import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class OneQubitGate(nn.Module):
    """Abstract Base Class for one qubit gates"""

    def __init__(self, number_of_parameters: int, params: torch.tensor = None):
        super().__init__()
        
        self.register_buffer("id", torch.eye(2, dtype=torch.cfloat, requires_grad=False))
        self.register_buffer("x", torch.tensor([[0, 1],
                                                [1, 0]], dtype=torch.cfloat, requires_grad=False))
        self.register_buffer("y", torch.tensor([[0, -1j],
                                                [1j, 0]], dtype=torch.cfloat, requires_grad=False))
        self.register_buffer("z", torch.tensor([[1, 0],
                                                [0, -1]], dtype=torch.cfloat, requires_grad=False))

        self.number_of_parameters = number_of_parameters
        if params is None:
            params = torch.zeros(self.number_of_parameters, dtype=float, requires_grad=True)
        self.params = torch.nn.Parameter(params.clone().detach().requires_grad_(True))
        assert self.params.shape == (
            self.number_of_parameters,), f"Mismatch in number_of_params and shape of params provided: {self.number_of_parameters} vs {self.params.shape[0]} "

    @torch.no_grad()
    def randomize_params(self):
        self.params = torch.nn.Parameter(torch.randn_like(self.params).requires_grad_(True))
    

class NMROneQubitGate(OneQubitGate):
    """One qubit gate limited to rotations in XY plane"""
    
    def __init__(self, params: torch.tensor = None):
        super().__init__(2, params=params)
        
    def forward(self):
        return (torch.cos(self.params[0] / 2) * self.id - 1j * torch.sin(self.params[0] / 2) * (torch.cos(self.params[1]) * self.x + torch.sin(self.params[1]) * self.y))

    
class GeneralOneQubitGate(OneQubitGate):
    """One qubit gate limited to rotations in XY plane"""
    
    def __init__(self, params: torch.tensor = None):
        super().__init__(3, params=params)
        
    def forward(self):
        mat = torch.cos(self.params[0] / 2) * self.id - 1j * torch.sin(self.params[0] / 2) * self.y
        mat[1] *= torch.e**(1j * self.params[1])
        mat[...][1] *= torch.e**(1j * self.params[2])
        return mat
           