import numpy as np
import random
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod

from onequbitgates import OneQubitGate, GeneralOneQubitGate, NMROneQubitGate


class MultiQubitGate(ABC):
    """Abstract Base Class for multi qubit gates"""

    def __init__(self, size: int, time: float = 0):
        self.time: float = time
        self.size: int = size  # number of qubits
        self.matrix: np.ndarray = np.ones((2 ** self.size, 2 ** self.size), dtype=complex)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def randomize_params(self):
        pass

    @abstractmethod
    def to_circuit(self):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class Delay(MultiQubitGate):
    """Evolution under fixed NMR Hamiltonian"""

    def __init__(self, size: int, time: float = 0):
        super().__init__(size, time)
        self._id = np.eye(2, dtype=complex)
        self._z = np.asarray([[1, 0],
                              [0, -1]], dtype=complex)

        self.j = np.zeros((self.size, self.size), dtype=float)  # interaction matrix
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                self.j[i][j] = 1
        self.hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
        self.update_hamiltonian()

        self.derivative = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)

    def sigma_z(self, i: int, j: int):
        matrix = np.ones((1, 1), dtype=complex)
        for k in range(self.size):
            if k == i or k == j:
                matrix = np.kron(matrix, self._z)
            else:
                matrix = np.kron(matrix, self._id)
        return matrix

    def update_hamiltonian(self):
        hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                hamiltonian += np.pi / 2 * self.j[i][j] * self.sigma_z(i, j)
        self.hamiltonian = hamiltonian

    def update_matrix(self):  # evolution matrix
        evolution = np.eye(2 ** self.size, dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                evolution = evolution @ (np.cos(np.pi / 2 * self.j[i][j] * self.time) * np.eye(2 ** self.size,
                                                                                               dtype=complex) - 1j * np.sin(
                    np.pi / 2 * self.j[i][j] * self.time) * self.sigma_z(i, j))
        self.matrix = evolution

    def update_derivative(self):
        self.derivative = -1j * self.hamiltonian @ self.matrix

    def update(self):
        self.update_hamiltonian()
        self.update_matrix()
        self.update_derivative()

    def set_j(self, new_j):
        for i in range(self.j.shape[0]):
            for j in range(self.j.shape[1]):
                self.j[i][j] = new_j[i][j]

    def randomize_params(self, exp=0.3):
        self.time = random.uniform(0, exp / 0.00148)

    def to_circuit(self):
        circuit = QuantumCircuit(self.size)
        circuit.hamiltonian(self.hamiltonian, float(self.time), circuit.qubits)
        return circuit

    def normalize(self):
        pass

    def __repr__(self):
        return f"{self.__class__} {self.time} {self.j}"


class Pulse(MultiQubitGate):
    """Concurrent one qubit gates on each qubit"""

    def __init__(self, size: int, params=None):
        super().__init__(size=size, time=0)

        if params is None:
            params = [[0, 0] for _ in range(self.size)]
        self.basic_gates = [NMROneQubitGate(param) for param in params]
        self.correction = [[0, 0] for _ in range(self.size)]
        self.derivative = np.zeros((self.size, 2, 2 ** self.size, 2 ** self.size), dtype=complex)

    def update_matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            matrix = np.kron(self.basic_gates[i].matrix, matrix)
        self.matrix = matrix

    def update_derivative(self):
        for qubit in range(self.size):
            for parameter in [0, 1]:
                qubit_parameter_derivative = np.ones(1, dtype=complex)
                for j in range(self.size):
                    if j != qubit:
                        qubit_parameter_derivative = np.kron(self.basic_gates[j].matrix, qubit_parameter_derivative)
                    else:
                        qubit_parameter_derivative = np.kron(self.basic_gates[j].derivative[parameter],
                                                             qubit_parameter_derivative)
                self.derivative[qubit][parameter] = qubit_parameter_derivative

    def update(self):
        for basic_gate in self.basic_gates:
            basic_gate.update()
        self.update_matrix()
        self.update_derivative()

    def randomize_params(self):
        for basicGate in self.basic_gates:
            basicGate.randomize_params()

    def to_circuit(self):
        circuit = QuantumCircuit(self.size)
        for i in range(self.size):
            circuit.r(self.basic_gates[i].params[0], self.basic_gates[i].params[1], circuit.qubits[i])
        return circuit

    def normalize(self):
        for basic_gate in self.basic_gates:
            basic_gate.normalize()

    def __repr__(self):
        string= self.__class__
        for i in range(self.size):
            string += " "
            string += self.basic_gates[i].__repr__()
        return string


class Inversion(MultiQubitGate):
    def __init__(self, size: int, qubits=None):
        super().__init__(size=size, time=0)
        self._id = np.eye(2, dtype=complex)
        self._x = np.asarray([[0, 1],
                              [1, 0]], dtype=complex)

        if qubits is None:
            qubits = []
        self.qubits = qubits
        self.derivative = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)

    def update_matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            if i in self.qubits:
                matrix = np.kron(self._x, matrix)
            else:
                matrix = np.kron(self._id, matrix)
        self.matrix = matrix

    def update(self):
        self.update_matrix()

    def randomize_params(self):
        pass

    def to_circuit(self):
        circuit = QuantumCircuit(self.size)
        for i in self.qubits:
            circuit.x(i)
        return circuit

    def normalize(self):
        pass

    def __repr__(self):
        string = self.__class__
        string += str(self.qubits)
        return string


class CXCascade(MultiQubitGate):
    """Cascade of CX gates in a closed loop"""

    def __init__(self, size: int):
        super().__init__(size=size)
        self._cx = np.asarray([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]], dtype=complex)
        self.construct_matrix()

    def apply_cx(self, i: int, j: int):
        a = np.array([[1]])
        b = np.array([[1]])
        for k in range(self.size):
            if k == i:
                a = np.kron(a, np.array([1, 0, 0, 0]).reshape(2, 2))
                b = np.kron(b, np.array([0, 0, 0, 1]).reshape(2, 2))
            else:
                if k == j:
                    a = np.kron(a, np.array([1, 0, 0, 1]).reshape(2, 2))
                    b = np.kron(b, np.array([0, 1, 1, 0]).reshape(2, 2))
                else:
                    a = np.kron(a, np.array([1, 0, 0, 1]).reshape(2, 2))
                    b = np.kron(b, np.array([1, 0, 0, 1]).reshape(2, 2))

        self.matrix = (a + b) @ self.matrix

    def construct_matrix(self):
        self.matrix = np.eye(2 ** self.size, dtype=complex)
        for i in range(self.size - 1):
            self.apply_cx(i, i + 1)
        self.apply_cx(self.size - 1, 0)

    def update(self):
        pass

    def randomize_params(self):
        pass

    def to_circuit(self):
        circuit = QuantumCircuit(self.size)
        for i in range(self.size - 1):
            circuit.cx(i, i + 1)
        circuit.cx(self.size - 1, 0)
        return circuit

    def normalize(self):
        pass

    def __repr__(self):
        return self.__class__
