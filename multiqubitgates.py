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
        self.matrix: np.ndarray = np.eye(2 ** self.size, dtype=complex)
        self.params_getter = None
        self.derivative = np.zeros((0, 2 ** self.size, 2 ** self.size), dtype=complex)

    @property
    def params(self):
        if self.params_getter is None:
            raise NotImplementedError("Params getter should be implemented and set")
        return self.params_getter

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

        self.derivative = np.zeros((1, 2 ** self.size, 2 ** self.size), dtype=complex)

        class ParamsGetter:
            def __init__(self, master):
                self.master = master

            def __getitem__(self, key: int):
                if key != 0:
                    raise KeyError
                return self.master.time

            def __setitem__(self, key: int, value: float):
                if key != 0:
                    raise KeyError
                self.master.time = value

            def __len__(self):
                return 1

        self.params_getter = ParamsGetter(self)

        self.update()

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
        self.derivative[0] = -1j * self.hamiltonian @ self.matrix

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
        return f"{self.__class__.__name__} {self.time} {self.j}"


class Pulse(MultiQubitGate):
    """Concurrent one qubit gates on each qubit"""

    def __init__(self, size: int, one_qubit_gate_type: type(OneQubitGate) = None, params=None):
        super().__init__(size=size, time=0)

        if one_qubit_gate_type is None:
            one_qubit_gate_type = GeneralOneQubitGate

        assert issubclass(one_qubit_gate_type, OneQubitGate), "one_qubit_gate_type must be OneQubitGate"

        if params is None:
            params = [None] * self.size
        self.basic_gates = [one_qubit_gate_type(param) for param in params]
        self.derivative = np.zeros((self.size * self.basic_gates[0].number_of_parameters, 2 ** self.size, 2 ** self.size), dtype=complex)

        class ParamsGetter:
            def __init__(self, master):
                self.master = master

            def __getitem__(self, key):
                # TODO: write exceptions for bad keys
                return self.master.basic_gates[key // self.master.basic_gates[0].number_of_parameters].params[
                    key % self.master.basic_gates[0].number_of_parameters]

            def __setitem__(self, key, value: float):
                # TODO: write exceptions for bad keys
                self.master.basic_gates[key // self.master.basic_gates[0].number_of_parameters].params[
                    key % self.master.basic_gates[0].number_of_parameters] = value

            def __len__(self):
                return len(self.master.basic_gates) * len(self.master.basic_gates[0].params)

        self.params_getter = ParamsGetter(self)
        self.update()

    def update_matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            matrix = np.kron(self.basic_gates[i].matrix, matrix)
        self.matrix = matrix

    def update_derivative(self):
        for qubit in range(self.size):
            for parameter in range(self.basic_gates[0].number_of_parameters):
                qubit_parameter_derivative = np.ones(1, dtype=complex)
                for j in range(self.size):
                    if j != qubit:
                        qubit_parameter_derivative = np.kron(self.basic_gates[j].matrix, qubit_parameter_derivative)
                    else:
                        qubit_parameter_derivative = np.kron(self.basic_gates[j].derivative[parameter],
                                                             qubit_parameter_derivative)
                self.derivative[qubit * self.basic_gates[0].number_of_parameters + parameter] = qubit_parameter_derivative

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
        string= self.__class__.__name__
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

        class ParamsGetter:
            def __init__(self, master):
                self.master = master

            def __getitem__(self, key):
                raise KeyError(f"{self.master.__class__.__name__} has no parameters")

            def __setitem__(self, key, value: float):
                raise KeyError(f"{self.master.__class__.__name__} has no parameters")

            def __len__(self):
                return 0

        self.params_getter = ParamsGetter(self)
        self.update()

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
        string = self.__class__.__name__
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

        class ParamsGetter:
            def __init__(self, master):
                self.master = master

            def __getitem__(self, key):
                raise KeyError(f"{self.master.__class__.__name__} has no parameters")

            def __setitem__(self, key, value: float):
                raise KeyError(f"{self.master.__class__.__name__} has no parameters")

            def __len__(self):
                return 0

        self.params_getter = ParamsGetter(self)

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
        return self.__class__.__name__
