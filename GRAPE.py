import numpy as np
import math
from math import pi
import random
from qiskit import QuantumCircuit
from dataclasses import dataclass


class BasicGate:  # 1-qubit gate
    def __init__(self, params=None):
        if params is None:
            params = [0, 0]
        self.params: list[float] = params
        # Pauli matrices
        self._id = np.eye(2, dtype=complex)
        self._x = np.asarray([[0, 1], [1, 0]], dtype=complex)
        self._y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)

    @property
    def matrix(self):  # straightforward matrix representation
        matrix = math.cos(self.params[0] / 2) * self._id - 1j * math.sin(self.params[0] / 2) * (
                    math.cos(self.params[1]) * self._x + math.sin(self.params[1]) * self._y)

        return matrix

    def randomize_params(self):
        self.params = [2 * math.pi * random.random(), 2 * math.pi * random.random()]

    def correct_params(self, correction):  # update parameters based on passed corrections
        self.params[0] += correction[0]
        self.params[1] += correction[1]

    def normalize_angles(self):
        self.params[0] = self.params[0].real % (4 * pi)
        self.params[1] = self.params[1].real % (2 * pi)


class Gate:
    def __init__(self, size: int = 2, time: float = 0):
        self.time: float = time
        self.size: int = size  # number of qubits
        self._id = np.eye(2, dtype=complex)
        self._x = np.asarray([[0, 1], [1, 0]], dtype=complex)
        self._y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
        self._z = np.asarray([[1, 0], [0, -1]], dtype=complex)

    @property
    def matrix(self):
        raise NotImplementedError()

    def randomize_params(self):
        raise NotImplementedError()

    def correct_params(self, correction):
        raise NotImplementedError()

    def to_qiskit(self):
        raise NotImplementedError()

    def normalize(self):
        raise NotImplementedError()


class Evolution(Gate):
    def __init__(self, size: int = 2, time: float = 0):
        super().__init__(size, time)

        self.J = np.zeros((self.size, self.size), dtype=complex)  # interaction matrix
        for i in range(self.size - 1):
            self.J[i + 1][i] = 1

    def sigma_z(self, i: int, j: int):
        matrix = np.ones((1, 1), dtype=complex)
        for k in range(self.size):
            if k == i or k == j:
                matrix = np.kron(matrix, self._z)
            else:
                matrix = np.kron(matrix, self._id)
        return matrix

    @property
    def hamiltonian(self):
        hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                hamiltonian += math.pi / 2 * self.J[i][j] * self.sigma_z(i, j)
        return hamiltonian

    @property
    def matrix(self):  # evolution matrix
        evolution = np.eye(2 ** self.size, dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                evolution = evolution @ (math.cos(math.pi / 2 * self.J[i][j] * self.time) * np.eye(2 ** self.size,
                                                                                                   dtype=complex) - 1j * math.sin(
                    math.pi / 2 * self.J[i][j] * self.time) * self.sigma_z(i, j))
        return evolution

    def set_j(self, new_j):
        for i in range(self.J.shape[0]):
            for j in range(self.J.shape[1]):
                self.J[i][j] = new_j[i][j]

    def randomize_params(self):
        self.time = random.random()

    def correct_params(self, correction):
        self.time += correction

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size)
        circuit.hamiltonian(self.hamiltonian, float(self.time), circuit.qubits)
        return circuit

    def normalize(self):
        pass


class Kick(Gate):
    def __init__(self, size: int = 2, params=None):
        super().__init__(size=size, time=0)

        if params is None:
            params = [[0, 0] for _ in range(self.size)]
        self.basic_gates = [BasicGate(param) for param in params]

    @property
    def matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            matrix = np.kron(self.basic_gates[i].matrix, matrix)
        return matrix

    def randomize_params(self):
        for basicGate in self.basic_gates:
            basicGate.randomize_params()

    def correct_params(self, correction):  # update parameters based on passed corrections
        for i in range(self.size):
            self.basic_gates[i].correct_params(correction)

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size)
        for i in range(self.size):
            circuit.r(self.basic_gates[i].params[0], self.basic_gates[i].params[1], circuit.qubits[i])
        return circuit

    def normalize(self):
        for basic_gate in self.basic_gates:
            basic_gate.normalize_angles()


class Inversion(Gate):
    def __init__(self, size: int = 2, qubits = None):
        super().__init__(size=size, time=0)

        if qubits is None:
            qubits = []
        self.qubits = qubits

    @property
    def matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            if i in self.qubits:
                matrix = np.kron(self._x, matrix)
            else:
                matrix = np.kron(self._id, matrix)
        return matrix

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size)
        for i in self.qubits:
            circuit.x(i)
        return circuit

    def randomize_params(self):
        pass

    def correct_params(self, correction):
        pass

    def normalize(self):
        pass

@dataclass()
class Implementation:
    name: str
    angles: list[list[list[float, float]]]
    times: list[float]
    phase: float = 0

    def save(self):
        #                 qubit 1   ... ...   qubit k   time
        # kick 1:       theta; phi; ... ... theta; phi; time;
        # ...
        # ...
        # kick n:       theta; phi; ... ... theta; phi; time;
        # global phase:    phase;   ... ...    phase;   phase;
        #
        file = open(self.name, "w")
        for i in range(len(self.angles)):
            for j in range(len(self.angles[0])):
                file.write(f"{self.angles[i][j][0]}; {self.angles[i][j][1]}; ")
            file.write(f"{self.times[i]}\n")
        for i in range(2 * len(self.angles[0]) + 1):
            file.write(f"{self.phase}")
            if i != 2 * len(self.angles[0]):
                file.write("; ")
        file.write("\n")

    def load(self, name):
        data = np.genfromtxt(name, delimiter=";")
        self.angles = [[[data[i][2 * j + 0], data[i][2 * j + 1]] for j in range(data.shape[1] - 1)] for i in range(data.shape[0])]
        self.times = [data[i][-1] for i in range(data.shape[0])]
        self.phase = data[-1][0]


class GradientDescend:
    def __init__(self, target, n: int = 4, implementation: Implementation = None):
        self.target = target  # unitary to approximate

        if implementation is not None:
            raise NotImplementedError()
        else:
            self.__size = int(math.log2(self.target.size) / 2)  # number of qubits
            self.phase = 0  # global phase
            self.gates = [] # simultaneous gates
            for _ in range(n):
                self.gates += [Kick(size=self.__size)]
                self.gates += [Evolution(size=self.__size)]
            self.gates += [Kick(size=self.__size)]
        self.stepSize = 0.001  # gradient-to-change ration

    @property
    def time(self):  # total approximation time
        time = 0
        for gate in self.gates:
            time += gate.time
        return time.real

    @property
    def target_d(self):  # target dagger
        return self.target.conjugate().transpose()

    @property
    def matrix(self):  # approximation matrix
        matrix = np.eye(2 ** self.__size, dtype=complex)

        for gate in self.gates:
            matrix = gate.matrix @ matrix

        return matrix * math.e ** (1j * self.phase)

    @property
    def distance(self):  # Frobenius norm
        return ((self.matrix - self.target) @ (self.matrix - self.target).conjugate().transpose()).trace()

    def randomize_params(self):  # randomizes params for 1-qubit operations
        for gate in self.gates:
            gate.randomize_params()

    def gradient_step(self):
        delta = 0.001
        current_dist = self.distance
        new_dist = self.distance

        for gate in self.gates:
            if type(gate) is Evolution:
                gate.time += delta
                new_dist = self.distance
                gate.time -= delta
                gate.time += (current_dist - new_dist) / delta * self.stepSize
            if type(gate) is Kick:
                for qubit in range(self.__size):
                    for parameter in [0, 1]:
                        gate.basic_gates[qubit].params[parameter] += delta
                        new_dist = self.distance
                        gate.basic_gates[qubit].params[parameter] -= delta
                        gate.basic_gates[qubit].params[parameter] += (current_dist - new_dist) / delta * self.stepSize
        self.phase -= np.angle((self.matrix @ self.target_d).trace())

    def descend(self, steps=1000, track_distance=False):  # perform gradient descent
        distances = []  # distances to track

        for i in range(steps):
            distances += [self.distance]
            self.gradient_step()

        # most parameters are cyclic - make them in (0, max)
        for gate in self.gates:
            gate.normalize()

        self.phase = self.phase.real % (2 * math.pi)

        if track_distance:
            return distances

    def to_qiskit(self):
        circuit = QuantumCircuit(self.__size, global_phase=self.phase)
        for gate in self.gates:
            circuit += gate.to_qiskit()
        return circuit

    def set_j(self, new_j):
        for gate in self.gates:
            if type(gate) is Evolution:
                gate.set_j(new_j)

    def make_times_positive(self):
        if self.__size not in [1, 2, 3]:
            raise NotImplementedError("Making time positive only possible for 1, 2 and 3 qubit systems")
        all_positive = False
        while not all_positive:
            for i in range(len(self.gates)):
                if type(self.gates[i]) is Evolution and self.gates[i].time < 0:
                    if type(self.gates[i + 1]) is Inversion:
                        self.gates[i].time *= -1
                        self.gates.pop(i + 1)
                        self.gates.pop(i - 1)
                        break
                    if type(self.gates[i + 1]) is Kick:
                        self.gates[i].time *= -1
                        self.gates.insert(i + 1, Inversion(self.__size, [1]))
                        self.gates.insert(i - 1, Inversion(self.__size, [1]))
                        break
                if i == len(self.gates) - 1:
                    all_positive = True


