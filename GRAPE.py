import numpy as np
import math
from math import pi
import random
from qiskit import QuantumCircuit


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

    def set_correction(self, correction):
        raise NotImplementedError()

    def correct_params(self):
        raise NotImplementedError()

    def to_qiskit(self):
        raise NotImplementedError()

    def normalize(self):
        raise NotImplementedError()

    def fake_circuit(self):
        raise NotImplementedError()


class Delay(Gate):
    def __init__(self, size: int = 2, time: float = 0):
        super().__init__(size, time)

        self.j = np.zeros((self.size, self.size), dtype=complex)  # interaction matrix
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                self.j[i][j] = 1
        self.correction = 0

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
                hamiltonian += math.pi / 2 * self.j[i][j] * self.sigma_z(i, j)
        return hamiltonian

    @property
    def matrix(self):  # evolution matrix
        evolution = np.eye(2 ** self.size, dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                evolution = evolution @ (math.cos(math.pi / 2 * self.j[i][j] * self.time) * np.eye(2 ** self.size,
                                                                                                   dtype=complex) - 1j * math.sin(
                    math.pi / 2 * self.j[i][j] * self.time) * self.sigma_z(i, j))
        return evolution

    def set_j(self, new_j):
        for i in range(self.j.shape[0]):
            for j in range(self.j.shape[1]):
                self.j[i][j] = new_j[i][j]

    def randomize_params(self):
        self.time = random.random()

    def set_correction(self, correction):
        self.correction = correction

    def correct_params(self):
        self.time += self.correction

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size)
        circuit.hamiltonian(self.hamiltonian, float(self.time), circuit.qubits)
        return circuit

    def normalize(self):
        pass

    def fake_circuit(self):
        return self.to_qiskit()


class Pulse(Gate):
    def __init__(self, size: int = 2, params=None):
        super().__init__(size=size, time=0)

        if params is None:
            params = [[0, 0] for _ in range(self.size)]
        self.basic_gates = [BasicGate(param) for param in params]
        self.correction = [[0, 0] for _ in range(self.size)]

    @property
    def matrix(self):
        matrix = np.ones(1, dtype=complex)
        for i in range(self.size):
            matrix = np.kron(self.basic_gates[i].matrix, matrix)
        return matrix

    def randomize_params(self):
        for basicGate in self.basic_gates:
            basicGate.randomize_params()

    def set_correction(self, correction):
        for i in range(self.size):
            self.correction[i][0] = correction[i][0]
            self.correction[i][1] = correction[i][1]

    def correct_params(self):  # update parameters based on passed corrections
        for i in range(self.size):
            self.basic_gates[i].correct_params(self.correction[i])

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size)
        for i in range(self.size):
            circuit.r(self.basic_gates[i].params[0], self.basic_gates[i].params[1], circuit.qubits[i])
        return circuit

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
    def __init__(self, size: int = 2, qubits=None):
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

    def set_correction(self, correction):
        pass

    def correct_params(self):
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
            self.__size = int(math.log2(self.target.size) / 2)  # number of qubits
            self.phase = 0  # global phase
            self.gates = []  # simultaneous gates
            for _ in range(n):
                self.gates += [Pulse(size=self.__size)]
                self.gates += [Delay(size=self.__size)]
            self.gates += [Pulse(size=self.__size)]
        self.stepSize = 0.01  # gradient-to-change ration
        self.noise = 0

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

    @property
    def j(self):
        for gate in self.gates:
            if type(gate) is Delay:
                return gate.j

    def randomize_params(self):  # randomizes params for 1-qubit operations
        for gate in self.gates:
            gate.randomize_params()

    def gradient_step(self):
        delta = 0.001
        current_dist = self.distance
        new_dist = self.distance

        gradient = [[] for _ in self.gates]

        for gate in self.gates:
            if type(gate) is Delay:
                gate.time += delta
                new_dist = self.distance
                gate.time -= delta
                gate.set_correction((current_dist - new_dist) / delta * self.stepSize)
            if type(gate) is Pulse:
                correction = [[0, 0] for _ in range(gate.size)]
                for qubit in range(self.__size):
                    for parameter in [0, 1]:
                        gate.basic_gates[qubit].params[parameter] += delta
                        new_dist = self.distance
                        gate.basic_gates[qubit].params[parameter] -= delta
                        correction[qubit][parameter] = (current_dist - new_dist) / delta * self.stepSize
                        correction[qubit][parameter] *= (1 + random.uniform(-self.noise, self.noise))
                gate.set_correction(correction)

        for gate in self.gates:
            gate.correct_params()
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
            if type(gate) is Delay:
                gate.set_j(new_j)

    def make_times_positive(self):
        if self.__size not in [1, 2, 3]:
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
                        self.gates.insert(i + 1, Inversion(self.__size, [1]))
                        self.gates.insert(i, Inversion(self.__size, [1]))
                        break
                if i == len(self.gates) - 1:
                    all_positive = True

    def to_text(self, filename=None):
        str = f"{self.__size} {self.phase}\n"
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
        for i in range(self.__size):
            for j in range(self.__size):
                str += f"{any_evolution.J[i][j].real} "
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
        self.__size = int(lines[0].split()[0])
        self.phase = float(lines[0].split()[1])
        for i in range(1, len(lines)):
            data = lines[i].split()
            if data[0] == "delay" or data[0] == "Evolution":
                self.gates += [Delay(time=float(data[1]))]
            if data[0] == "pulse" or data[0] == "Kick":
                params = []
                for j in range(self.__size):
                    params += [[float(data[2 * j + 1]), float(data[2 * j + 2])]]
                self.gates += [Pulse(params=params)]
            if data[0] == "inversion" or data[0] == "Inversion":
                self.gates += [Inversion(qubits=[int(qubit) for qubit in data[1:]])]
            if data[0] == "J":
                j = np.asarray(data[1:], dtype=float).reshape((self.__size, self.__size))
                self.set_j(j)
        file.close()

    def print_times(self):
        for gate in self.gates:
            print(gate.time, end=" ")
        print("\n")

    def to_device_text(self, filename=None):
        if self.__size != 2:
            raise NotImplementedError()
        str = f"{self.__size} {self.phase}\n"
        any_evolution = None
        for gate in self.gates:
            if type(gate) is Delay:
                any_evolution = gate
                str += f"Evolution {gate.time.real * gate.J[0][1].real} \n"
            if type(gate) is Pulse:
                str += "Kick "
                for basic_gate in gate.basic_gates:
                    str += f"{basic_gate.params[0].real} {basic_gate.params[1].real} "
                str += "\n"
            if type(gate) is Inversion:
                str += "Kick "
                for qubit in range(self.__size):
                    if qubit in gate.qubits:
                        str += f"{math.pi} {0} "
                    else:
                        str += "0 0 "
                str += "\n"

        str += "J "
        for i in range(self.__size):
            for j in range(self.__size):
                str += f"{any_evolution.J[i][j].real} "
        str += "\n"

        if filename is not None:
            file = open(filename, "w")
            file.write(str)
            file.close()
            return
        else:
            return str


class ThreeQubitGradient(GradientDescent):
    def __init__(self, target, n: int = 4, filename=None):
        super().__init__(target=target, n=n, filename=filename)
        # if self.__size != 3:
        #     raise NotImplementedError(f"Using this class to implement {self.__size} qubit scheme when 3 is assumed")

    def hyper_step(self):  # this is needed to overcome the weakness of 0-2 qubit interaction
        j = self.j
        time_changes = []

        for gate in self.gates:
            if type(gate) is Delay:
                d = self.distance
                gate.time += 4 / j[1][2]
                d_up = self.distance
                gate.time -= 8 / j[1][2]
                d_down = self.distance
                gate.time += 4 / j[1][2]

                p_up = math.e ** (- (d_up - d) / 16 * 3)
                if np.random.binomial(1, p_up):
                    time_changes += [4 / j[1][2]]
                else:
                    time_changes += [0]

                p_down = math.e ** (- (d_down - d) / 16 * 3)
                if np.random.binomial(1, p_down):
                    time_changes[-1] += 4 / j[1][2]
                else:
                    time_changes[-1] += 0

        i = 0
        for gate in self.gates:
            if type(gate) is Delay:
                gate.time += time_changes[i]
                i += 1

