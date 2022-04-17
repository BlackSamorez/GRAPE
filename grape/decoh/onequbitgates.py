import torch
import torch.nn as nn

import math

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
    

    
class GeneralOneQubitGate(OneQubitGate):
    """One qubit gate limited to rotations in XY plane"""
    
    def __init__(self, params: torch.tensor = None):
        super().__init__(3, params=params)
        self.t = 35.6     # ns
        self.T1 = 46.4e3  # ns
        self.T2 = 105e3   # ns
        
        self.lambda_A = 0.1  # 1 - torch.e**(-self.t/self.T1)
        self.lambda_p = 0.2  # 1 - torch.e**(-self.t/self.T2)
        
        self.register_buffer("decoh_matrix", torch.diag(torch.tensor([1-self.lambda_A,
                                                                      math.sqrt((1-self.lambda_A)*(1-self.lambda_p)),
                                                                      math.sqrt((1-self.lambda_A)*(1-self.lambda_p)),
                                                                      1-self.lambda_A], dtype=torch.cfloat, requires_grad=False)))
        self.register_buffer("decoh_offset", torch.tensor([self.lambda_A, 0, 0, 0], dtype=torch.cfloat, requires_grad=False))
        
    def apply_decoh(self, rho):
        return (self.decoh_matrix @ rho.reshape(4) + self.decoh_offset).reshape(2, 2)
    
    def apply_rx(self, rho, angle):
        rx = torch.cos(angle/2) * self.id - 1j * torch.sin(angle/2) * self.x
        return rx @ rho @ rx.T.conj()
    
    def apply_rz(self, rho, angle):
        rz = torch.cos(angle/2) * self.id - 1j * torch.sin(angle/2) * self.z
        return rz @ rho @ rz.T.conj()
        
    def forward(self, rho):  # x is \theta, \phi
        rho = self.apply_rz(rho, self.params[2])
        rho = self.apply_rx(rho, torch.tensor([torch.pi/2], dtype=torch.cfloat, requires_grad=False))
        rho = self.apply_decoh(rho)
        rho = self.apply_rz(rho, self.params[1])
        rho = self.apply_rx(rho, torch.tensor([-torch.pi/2], dtype=torch.cfloat, requires_grad=False))
        rho = self.apply_decoh(rho)
        rho = self.apply_rz(rho, self.params[0])
        
        return rho
    
           