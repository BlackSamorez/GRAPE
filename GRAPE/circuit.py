import numpy as np
from .multiqubitgates import MultiQubitGate, Pulse
from .onequbitgates import NMROneQubitGate


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
        self.matrix = matrix

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
        if issubclass(type(other), MultiQubitGate):
            self.gates.append(other)
            return self
        raise TypeError("Other must be Architecture or MultiQubitGate")

    def __add__(self, other):
        result = Circuit(self.size)
        for i in range(len(self)):
            result.gates.append(self.gates[i])
        if type(other) is type(self):
            assert other.size == self.size, "other must be of same size as self"
            for i in range(len(other.gates)):
                result += other.gates[i]
            return result
        if issubclass(other.__class__, MultiQubitGate):
            if other.size != self.size:
                raise NotImplementedError
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

    def derivative(self, derivative_gate, parameter: int):
        if type(derivative_gate) is int:
            derivative_gate = self.gates[derivative_gate]

        if not issubclass(type(derivative_gate), MultiQubitGate):
            raise TypeError(f"gate argument must be of int or MuliQubitGate type. {derivative_gate.__class__.__name__} was given")

        derivative = np.eye(2 ** self.size, dtype=complex)
        for gate in self.gates:
            if gate is derivative_gate:
                derivative = gate.derivative[parameter] @ derivative
            else:
                derivative = gate.matrix @ derivative

        return derivative

    def normalize(self):
        for gate in self.gates:
            gate.normalize()


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
            self.gates.append(Pulse(size, one_qubit_gate_type=NMROneQubitGate))
