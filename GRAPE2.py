import numpy as np
import math
from math import pi
import random
from qiskit import QuantumCircuit


# Обновление после изменения со дна наверх
class BasicDensityGate:  # 1-qubit gate for density matrices
    def __init__(self, params=None):
        if params is None:
            params = [0, 0]
        self.params: list[float] = params
        self.matrix = np.ones(4, dtype=np.complex)
        self.derivative = [np.zeros((4, 4), dtype=np.complex) for _ in range(2)]

    def update_matrix(self):  # straightforward matrix representation
        self.matrix = np.asarray([[math.cos(self.params[0] / 2) ** 2 + (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2) ** 2, 0, 0, 0],
                                  [0, math.cos(self.params[0] / 2) ** 2 + (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2 / 2 + (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2 / 2, 0.5j * (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2 + -0.5j * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2, 1j * math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2) + -1j * math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2)],
                                  [0, 0.5j * (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2 + -0.5j * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2, math.cos(self.params[0] / 2) ** 2 - (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2 / 2 - (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) ** 2 * math.sin(self.params[0] / 2) ** 2 / 2, -math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2) - math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2)],
                                  [0, -1j * math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2) + 1j * math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2), math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2) + math.cos(self.params[0] / 2) * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2), math.cos(self.params[0] / 2) ** 2 - (math.cos(self.params[1]) + -1j * math.sin(self.params[1])) * (math.cos(self.params[1]) + 1j * math.sin(self.params[1])) * math.sin(self.params[0] / 2) ** 2]], dtype=complex)

    def update_derivative(self):
        d_theta = -1 / 2 * math.sin(self.params[0] / 2) * self._id - 1j / 2 * math.cos(self.params[0] / 2) * (
                    math.cos(self.params[1]) * self._x + math.sin(self.params[1]) * self._y)
        d_phi = 1j * math.sin(self.params[0] / 2) * (
                    math.sin(self.params[1]) * self._x + math.cos(self.params[1]) * self._y)
        self.derivative = [d_theta, d_phi]

    def update(self):
        self.update_matrix()
        self.update_derivative()

    def randomize_params(self):
        self.params = [2 * math.pi * random.random(), 2 * math.pi * random.random()]

    def normalize_angles(self):  # no update needed
        self.params[0] = self.params[0].real % (4 * pi)
        self.params[1] = self.params[1].real % (2 * pi)


class DensityGate:
    def __init__(self, size: int, time: float = 0):
        self.time: float = time
        self.size: int = size  # number of qubits
        self.matrix = np.zeros((4 ** self.size, 4 ** self.size), dtype=complex)

    def update(self):
        raise NotImplementedError()

    def randomize_params(self):
        raise NotImplementedError()

    def normalize(self):
        raise NotImplementedError()

    def fake_circuit(self):
        raise NotImplementedError()


# class Delay(Gate):
#     def __init__(self, size: int, time: float = 0):
#         super().__init__(size, time)
#
#         self.j = np.zeros((self.size, self.size), dtype=complex)  # interaction matrix
#         for i in range(self.size - 1):
#             for j in range(i + 1, self.size):
#                 self.j[i][j] = 1
#         self.hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
#         self.update_hamiltonian()
#
#         self.derivative = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
#
#     def sigma_z(self, i: int, j: int):
#         matrix = np.ones((1, 1), dtype=complex)
#         for k in range(self.size):
#             if k == i or k == j:
#                 matrix = np.kron(matrix, self._z)
#             else:
#                 matrix = np.kron(matrix, self._id)
#         return matrix
#
#     def update_hamiltonian(self):
#         hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
#         for i in range(self.size):
#             for j in range(i + 1, self.size):
#                 hamiltonian += math.pi / 2 * self.j[i][j] * self.sigma_z(i, j)
#         self.hamiltonian = hamiltonian
#
#     def update_matrix(self):  # evolution matrix
#         evolution = np.eye(2 ** self.size, dtype=complex)
#         for i in range(self.size):
#             for j in range(i + 1, self.size):
#                 evolution = evolution @ (math.cos(math.pi / 2 * self.j[i][j] * self.time) * np.eye(2 ** self.size,
#                                                                                                    dtype=complex) - 1j * math.sin(
#                     math.pi / 2 * self.j[i][j] * self.time) * self.sigma_z(i, j))
#         self.matrix = evolution
#
#     def update_derivative(self):
#         self.derivative = -1j * self.hamiltonian @ self.matrix
#
#     def update(self):
#         self.update_hamiltonian()
#         self.update_matrix()
#         self.update_derivative()
#
#     def set_j(self, new_j):
#         for i in range(self.j.shape[0]):
#             for j in range(self.j.shape[1]):
#                 self.j[i][j] = new_j[i][j]
#
#     def randomize_params(self, exp=0.3):
#         self.time = random.uniform(0, exp / 0.00148)
#
#     def normalize(self):
#         pass
#
#     def fake_circuit(self):
#         return self.to_qiskit()


class Pulse(DensityGate):
    def __init__(self, size: int, params=None):
        super().__init__(size=size, time=0)

        if params is None:
            params = [[0, 0] for _ in range(self.size)]
        self.basic_gates = [BasicDensityGate(param) for param in params]
        self.correction = [[0, 0] for _ in range(self.size)]
        self.derivative = [[np.zeros((4 ** self.size, 4 ** self.size), dtype=complex),
                            np.zeros((4 ** self.size, 4 ** self.size), dtype=complex)] for _ in range(self.size)]

    def update_matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            matrix = np.kron(self.basic_gates[i].matrix, matrix)
        self.matrix = matrix

    def update_derivative(self):
        for qubit in range(self.size):
            for parameter in [0, 1]:
                derivative = np.ones(1, dtype=complex)
                for j in range(self.size):
                    if j != qubit:
                        derivative = np.kron(self.basic_gates[j].matrix, derivative)
                    else:
                        derivative = np.kron(self.basic_gates[j].derivative[parameter], derivative)
                self.derivative[qubit][parameter] = derivative

    def update(self):
        for basic_gate in self.basic_gates:
            basic_gate.update()
        self.update_matrix()
        self.update_derivative()

    def randomize_params(self):
        for basicGate in self.basic_gates:
            basicGate.randomize_params()

    def normalize(self):
        for basic_gate in self.basic_gates:
            basic_gate.normalize_angles()

    def fake_circuit(self):
        circuit = QuantumCircuit(self.size)
        for i in range(self.size):
            circuit.r(self.basic_gates[i].params[0] * 2 / math.pi, self.basic_gates[i].params[1] * 180 / math.pi,
                      circuit.qubits[i])
        return circuit


class Inversion(Gate):
    def __init__(self, size: int, qubits=None):
        super().__init__(size=size, time=0)

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

    def normalize(self):
        pass

    def fake_circuit(self):
        return self.to_qiskit()


class GradientDescent:
    def __init__(self, target, n: int = 4, filename=None):
        self.target = target  # unitary to approximate

        if filename is not None:
            self.read_text(filename)
        else:
            self._size = int(math.log2(self.target.size) / 2)  # number of qubits
            self.phase = 0  # global phase
            self.gates = []  # simultaneous gates
            for _ in range(n):
                self.gates += [Pulse(size=self._size)]
                self.gates += [Delay(size=self._size)]
            self.gates += [Pulse(size=self._size)]
        self.stepSize = 0.01  # gradient-to-change ration
        self.noise = 0
        self.matrix = np.ones((2 ** self._size, 2 ** self._size), dtype=complex)
        self.approx_time = 300
        self.update()

    @property
    def time(self):  # total approximation time
        time = 0
        for gate in self.gates:
            time += gate.time
        return time.real

    @property
    def target_d(self):  # target dagger
        return self.target.conjugate().transpose()

    def update_matrix(self):  # approximation matrix
        matrix = np.eye(2 ** self._size, dtype=complex)
        for gate in self.gates:
            matrix = gate.matrix @ matrix
        self.matrix = matrix * math.e ** (1j * self.phase)

    @property
    def distance(self):  # Frobenius norm
        return ((self.matrix - self.target) @ (self.matrix - self.target).conjugate().transpose()).trace()

    @property
    def j(self):
        for gate in self.gates:
            if type(gate) is Delay:
                return gate.j

    def update(self):
        for gate in self.gates:
            gate.update()
        self.update_matrix()

    def randomize_params(self):  # randomizes params for 1-qubit operations
        for gate in self.gates:
            if type(gate) is Delay:
                gate.randomize_params(2 / len(self.gates))
            else:
                gate.randomize_params()

    def corrections_from_gradients(self, time_sensitive=False):
        for i in range(len(self.gates)):
            if type(self.gates[i]) is Delay:
                matrix = np.eye(2 ** self._size, dtype=complex)
                for j in range(len(self.gates)):
                    if i != j:
                        matrix = self.gates[j].matrix @ matrix
                    else:
                        matrix = self.gates[j].derivative @ matrix
                if not time_sensitive:
                    self.gates[i].time -= self.stepSize * (
                                ((self.matrix - self.target) @ matrix.conjugate().transpose()).trace() + (
                                    matrix @ (self.matrix - self.target).conjugate().transpose()).trace())
                else:
                    self.gates[i].time -= self.stepSize * (
                                ((self.matrix - self.target) @ matrix.conjugate().transpose()).trace() + (matrix @ (
                                    self.matrix - self.target).conjugate().transpose()).trace() + self.distance / self.approx_time) * math.e ** (
                                                      self.time / self.approx_time)
            if type(self.gates[i]) is Pulse:
                for qubit in range(self._size):
                    for parameter in [0, 1]:
                        matrix = np.eye(2 ** self._size, dtype=complex)
                        for j in range(len(self.gates)):
                            if i != j:
                                matrix = self.gates[j].matrix @ matrix
                            else:
                                matrix = self.gates[j].derivative[qubit][parameter] @ matrix
                        self.gates[i].basic_gates[qubit].params[parameter] -= self.stepSize * (
                                    ((self.matrix - self.target) @ matrix.conjugate().transpose()).trace() + (
                                        matrix @ (self.matrix - self.target).conjugate().transpose()).trace())

    def descend(self, steps=1000, track_distance=False, time_sensitive=False):
        distances = []  # distances to track

        for i in range(steps):
            distances += [self.distance]
            self.corrections_from_gradients(time_sensitive=time_sensitive)
            self.update()
            self.phase -= np.angle((self.matrix @ self.target_d).trace())

        # most parameters are cyclic - make them in (0, max)
        for gate in self.gates:
            gate.normalize()

        self.phase = self.phase.real % (2 * math.pi)

        if track_distance:
            return distances

    def make_times_positive(self):
        if self._size not in [1, 2, 3]:
            raise NotImplementedError("Making time positive only possible for 1, 2 and 3 qubit systems")
        all_positive = False
        while not all_positive:
            for i in range(len(self.gates)):
                if type(self.gates[i]) is Delay and self.gates[i].time < 0:
                    if type(self.gates[i + 1]) is Inversion:
                        # print(f"popped inversion at {i}")
                        self.gates[i].time *= -1
                        self.gates.pop(i + 1)
                        self.gates.pop(i - 1)
                        break
                    if type(self.gates[i + 1]) is Pulse:
                        # print(f"added inversion at {i}")
                        self.gates[i].time *= -1
                        self.gates.insert(i + 1, Inversion(self._size, [1]))
                        self.gates.insert(i, Inversion(self._size, [1]))
                        break
                if i == len(self.gates) - 1:
                    all_positive = True

    def to_text(self, filename=None):
        str = f"{self._size} {self.phase}\n"
        any_evolution = None
        for gate in self.gates:
            if type(gate) is Delay:
                any_evolution = gate
                str += f"delay {gate.time.real} \n"
            if type(gate) is Pulse:
                str += "pulse "
                for basic_gate in gate.basic_gates:
                    str += f"{basic_gate.params[0].real} {basic_gate.params[1].real} "
                str += "\n"
            if type(gate) is Inversion:
                str += "inversion "
                for qubit in gate.qubits:
                    str += f"{qubit} "
                str += "\n"

        str += "J "
        for i in range(self._size):
            for j in range(self._size):
                str += f"{any_evolution.j[i][j].real} "
        str += "\n"

        if filename is not None:
            file = open(filename, "w")
            file.write(str)
            file.close()
            return
        else:
            return str

    def read_text(self, filename):
        self.gates = []
        file = open(filename, "r")
        lines = file.readlines()
        self._size = int(lines[0].split()[0])
        self.phase = float(lines[0].split()[1])
        for i in range(1, len(lines)):
            data = lines[i].split()
            if data[0] == "delay" or data[0] == "Evolution":
                self.gates += [Delay(size=self._size, time=float(data[1]))]
            if data[0] == "pulse" or data[0] == "Kick":
                params = []
                for j in range(self._size):
                    params += [[float(data[2 * j + 1]), float(data[2 * j + 2])]]
                self.gates += [Pulse(size=self._size, params=params)]
            if data[0] == "inversion" or data[0] == "Inversion":
                self.gates += [Inversion(size=self._size, qubits=[int(qubit) for qubit in data[1:]])]
            if data[0] == "J":
                j = np.asarray(data[1:], dtype=float).reshape((self._size, self._size))
                self.set_j(j)
        file.close()

    def print_times(self):
        for gate in self.gates:
            print(gate.time.real, end=" ")
        print("\n")

    def to_device_text(self, filename=None):
        if self._size not in [2, 3]:
            raise NotImplementedError()
        str = f"{self._size} {self.phase}\n"
        any_evolution = None
        for gate in self.gates:
            if type(gate) is Delay:
                any_evolution = gate
                str += f"Evolution {gate.time.real * gate.j[0][1].real} \n"
            if type(gate) is Pulse:
                str += "Kick "
                for basic_gate in gate.basic_gates:
                    str += f"{basic_gate.params[0].real} {basic_gate.params[1].real} "
                str += "\n"
            if type(gate) is Inversion:
                str += "Kick "
                for qubit in range(self._size):
                    if qubit in gate.qubits:
                        str += f"{math.pi} {0} "
                    else:
                        str += "0 0 "
                str += "\n"

        str += "J "
        for i in range(self._size):
            for j in range(self._size):
                str += f"{any_evolution.j[i][j].real} "
        str += "\n"

        if filename is not None:
            file = open(filename, "w")
            file.write(str)
            file.close()
            return
        else:
            return str


def time(filename: str):
    total_time = 0
    file = open(filename, "r")
    lines = file.readlines()
    for i in range(1, len(lines)):
        data = lines[i].split()
        if data[0] == "delay" or data[0] == "Evolution":
            total_time += float(data[1])
    file.close()
    return total_time
