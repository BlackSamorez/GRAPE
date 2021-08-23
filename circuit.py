import numpy as np
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod

from multiqubitgates import MultiQubitGate, Delay, Pulse, Inversion, CXCascade


class Circuit:
    """Class for representing quantum circuits as a series of quantum gates"""
    def __init__(self, size: int):
        self.size = size
        self.gates: list[MultiQubitGate] = []
        self.matrix = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)

    def update(self):
        """Update matrices and derivatives of all gates"""
        for gate in self.gates:
            gate.update()
        matrix = np.eye(2 ** self.size, dtype=complex)
        for gate in self.gates:
            matrix = gate.matrix @ matrix
        self.matrix = maatrix

    def randomize_params(self):
        """Randomize circuit params"""
        for gate in self.gates:
            gate.randomize_params()

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

    @property
    def time(self):
        time = 0
        for gate in self.gates:
            time += gate.time
        return time


class OneQubitEntanglementAlternation(Circuit):
    """Quantum circuit consisting of alternating single qubit rotations cascades and entanglement gates"""
    def __init__(self, size: int, entanglement_gate_type: type(MultiQubitGate), number_of_entanglements: int):
        super().__init__(size)
        assert issubclass(entanglement_gate_type, MultiQubitGate), "entanglement_gate_type must be a subclass of " \
                                                                   "MultiQubitGate "
        if entanglement_gate_type is Pulse:
            raise Warning("Current architecture provides no qubit entanglement!")

        self.gates.append(Pulse(size))
        for i in range(number_of_entanglements):
            self.gates.append(entanglement_gate_type(size))
            self.gates.append(Pulse(size))
