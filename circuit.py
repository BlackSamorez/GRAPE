import numpy as np
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod

from multiqubitgates import MultiQubitGate, Delay, Pulse, Inversion, CXCascade


class Circuit:
    """Class for representing quantum circuits as a series of quantum gates"""
    def __init__(self):
        self.gates: list[MultiQubitGate] = []

    def __len__(self):
        return len(self.gates)

    def __iadd__(self, other):
        if type(other) is type(self):
            for i in range(len(other)):
                self.gates.append(other.gates[i])
            return self
        if type(other) is MultiQubitGate:
            self.gates.append(other)
            return self
        raise TypeError("Other must be Architecture or MultiQubitGate")

    def __add__(self, other):
        result = Circuit()
        for i in range(len(self)):
            result.gates.append(self.gates[i])
        if type(other) is type(self):
            for i in range(len(other.gates)):
                result += other.gates[i]
            return result
        if issubclass(other, MultiQubitGate):
            result += other
            return result
        raise TypeError("Other must be Architecture or MultiQubitGate")

    def __repr__(self):
        return ""

    def __str__(self):
        if len(self) == 0:
            return ""
        string = ""
        for i in range(len(self) - 1):
            string += self.gates[i].__name__
            string += " "
        return string[:-1]


class OneQubitEntanglementAlternation(Circuit):
    """Quantum circuit consisting of alternating single qubit rotations cascades and entanglement gates"""
    def __init__(self, entanglement_gate_type: type(MultiQubitGate), number_of_entanglements: int):
        super().__init__()
        assert issubclass(entanglement_gate_type, MultiQubitGate), "entanglement_gate_type must be a subclass of " \
                                                                   "MultiQubitGate "
        if entanglement_gate_type is Pulse:
            raise Warning("Current architecture provides no qubit entanglement!")

        self.gates.append(Pulse)
        for i in range(number_of_entanglements):
            self.gates.append(entanglement_gate_type)
            self.gates.append(Pulse)
