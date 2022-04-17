import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from .onequbitgates import OneQubitGate, NMROneQubitGate, GeneralOneQubitGate


class MultiQubitGate(nn.Module):
    """Abstract Base Class for multi qubit gates"""

    def __init__(self, size: int, time: float = 0):
        super().__init__()
        self.time: float = time
        self.size: int = size  # number of qubits

    @abstractmethod
    def randomize_params(self):
        pass




class Pulse(MultiQubitGate):
    """Concurrent one qubit gates on each qubit"""

    def __init__(self, size: int, one_qubit_gate_type=None):
        super().__init__(size=size, time=0)
        if type(one_qubit_gate_type) is str:
            # TODO: implement match-case when available
            one_qubit_gate_type = one_qubit_gate_type.lower()
            general_aliases = ["general", str(GeneralOneQubitGate.__class__.__name__).lower()]
            nmr_aliases = ["nmr", str(NMROneQubitGate.__class__.__name__).lower()]

            if one_qubit_gate_type in general_aliases:
                one_qubit_gate_type = GeneralOneQubitGate
            if one_qubit_gate_type in nmr_aliases:
                one_qubit_gate_type = NMROneQubitGate
            else:
                raise ValueError(
                    f"one_qubit_gate_type must be OneQubit gate type or string in {general_aliases + nmr_aliases}")

        if one_qubit_gate_type is None:
            one_qubit_gate_type = GeneralOneQubitGate

        assert issubclass(one_qubit_gate_type, OneQubitGate), "one_qubit_gate_type must be OneQubitGate"

        self.basic_gates = [one_qubit_gate_type() for _ in range(self.size)]
        
        for i, basic_gate in zip(range(len(self.basic_gates)), self.basic_gates):
            self.add_module(f"basic_gate_{i}", basic_gate)

    def forward(self, x):
        submatrices = [basic_gate() for basic_gate in self.basic_gates]
        matrix = torch.tensor(1, dtype=torch.cfloat, requires_grad=False, device=self.basic_gates[0].params.device)
        for i in range(self.size):
            matrix = torch.kron(submatrices[i], matrix)
        return matrix @ x

    def randomize_params(self):
        for basicGate in self.basic_gates:
            basicGate.randomize_params()

            
class CXCascade(MultiQubitGate):
    """Cascade of CX gates in a closed loop"""

    def __init__(self, size: int):
        super().__init__(size=size) 
        self.register_buffer("cx", torch.tensor([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 0, 1],
                                                 [0, 0, 1, 0]], dtype=torch.cfloat, requires_grad=False))
        self.register_buffer("cascade", self.construct_matrix())

    @torch.no_grad()
    def apply_cx(self, i: int, j: int):
        a = torch.tensor([[1]], dtype=torch.cfloat, device=self.cx.device)
        b = torch.tensor([[1]], dtype=torch.cfloat, device=self.cx.device)
        for k in range(self.size):
            if k == i:
                a = torch.kron(a, torch.tensor([1, 0, 0, 0], device=self.cx.device).reshape(2, 2))
                b = torch.kron(b, torch.tensor([0, 0, 0, 1], device=self.cx.device).reshape(2, 2))
            else:
                if k == j:
                    a = torch.kron(a, torch.tensor([1, 0, 0, 1], device=self.cx.device).reshape(2, 2))
                    b = torch.kron(b, torch.tensor([0, 1, 1, 0], device=self.cx.device).reshape(2, 2))
                else:
                    a = torch.kron(a, torch.tensor([1, 0, 0, 1], device=self.cx.device).reshape(2, 2))
                    b = torch.kron(b, torch.tensor([1, 0, 0, 1], device=self.cx.device).reshape(2, 2))

        return a + b

    @torch.no_grad()
    def construct_matrix(self):
        tensor = torch.eye(2 ** self.size, dtype=torch.cfloat, device=self.cx.device)
        for i in range(self.size - 1):
            tensor = self.apply_cx(i, i + 1) @ tensor
        tensor = self.apply_cx(self.size - 1, 0) @ tensor
        return tensor
    
    def forward(self, x):
        return self.cascade @ x

    def randomize_params(self):
        pass
    
    
class DummyInput(MultiQubitGate):
    def __init__(self, size: int):
        super().__init__(size=size, time=0)
        self.register_buffer("dummy_input", torch.eye(2 ** self.size, dtype=torch.cfloat, requires_grad=False))
        
    def forward(self, x):
        return self.dummy_input
