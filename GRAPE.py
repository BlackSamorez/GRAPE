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
        self.__Id = np.eye(2, dtype=complex)
        self.__X = np.asarray([[0, 1], [1, 0]], dtype=complex)
        self.__Y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)

    @property
    def matrix(self):  # straightforward matrix representation
        matrix = math.cos(self.params[0] / 2) * self.__Id - 1j * math.sin(self.params[0] / 2) * (
                    math.cos(self.params[1]) * self.__X + math.sin(self.params[1]) * self.__Y)

        return matrix

    def randomize_params(self):
        self.params = [2 * math.pi * random.random(), 2 * math.pi * random.random()]

    def correct_params(self, correction):  # update parameters based on passed corrections
        self.params[0] += correction[0]
        self.params[1] += correction[1]


class EvolutionStep:  # a combination of 1-qubit operations followed by evolution
    def __init__(self, size: int = 2, params=None, time: float = 0):
        if params is None:
            params = []
        self.size: int = size  # number of qubits
        self.time: float = time  # evolution time

        if len(params) != self.size:
            params += [[0, 0] for _ in range(self.size - len(params))]

        self.basicGates = [BasicGate(param) for param in params]  # 1-qubit gates

        self.J = np.zeros((self.size, self.size), dtype=complex)  # interaction matrix (NOT FINISHED)
        for i in range(self.size - 1):
            self.J[i + 1][i] = 1

        self.__Id = np.eye(2, dtype=complex)
        self.__Z = np.asarray([[1, 0], [0, -1]], dtype=complex)

    def sigmaz(self, i: int, j: int):
        matrix = np.ones((1, 1), dtype=complex)
        for k in range(self.size):
            if k == i or k == j:
                matrix = np.kron(matrix, self.__Z)
            else:
                matrix = np.kron(matrix, self.__Id)
        return matrix

    @property
    def hamiltonian(self):
        hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                hamiltonian += math.pi / 2 * self.J[i][j] * self.sigmaz(i, j)
        return hamiltonian

    @property
    def evolution(self):  # evolution matrix
        evolution = np.eye(2 ** self.size, dtype=complex)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                evolution = evolution @ (math.cos(math.pi / 2 * self.J[i][j] * self.time) * np.eye(2 ** self.size,
                                                                                                   dtype=complex) - 1j * math.sin(
                    math.pi / 2 * self.J[i][j] * self.time) * self.sigmaz(i, j))
        return evolution  # NOT FINISHED

    @property
    def matrix(self):  # unitary of this evolution step
        matrix = np.ones(1, dtype=complex)

        for i in range(self.size):
            matrix = np.kron(self.basicGates[i].matrix, matrix)  # 1-qubit "kick"
        matrix = self.evolution @ matrix  # evolution

        return matrix

    def randomize_params(self):
        for basicGate in self.basicGates:
            basicGate.randomize_params()

    def correct_params(self, correction, evolution_correction):  # update parameters based on passed corrections
        for i in range(self.size):
            self.basicGates[i].correct_params(correction[i])
        self.time += evolution_correction

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size)
        for i in range(self.size):
            circuit.r(self.basicGates[i].params[0], self.basicGates[i].params[1], circuit.qubits[i])
        circuit.hamiltonian(self.hamiltonian, float(self.time), circuit.qubits)
        return circuit
    
    def set_j(self, new_j):
        for i in range(self.J.shape[0]):
            for j in range(self.J.shape[1]):
                self.J[i][j] = new_j[i][j]


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


class Evolution:  # class to approximate abstract unitary using a series of evolutionStep
    def __init__(self, target, n: int = 4, implementation: Implementation = None):
        self.target = target  # unitary to approximate
        self.noise: float = 0.05  # noise levels (see self.stupidCalculateGradient)

        if implementation is not None:
            if len(target) != (2 ** len(implementation.angles[0])):
                print(target.size, (2 ** len(implementation.angles[0])))
                raise Exception('Mismatch in target and implementation dimensions!')
            self.__n = len(implementation.angles)
            self.__size = len(implementation.angles[0])
            self.phase = implementation.phase
            self.gates = [EvolutionStep(size=self.__size, params=implementation.angles[i], time=implementation.times[i]) for i in range(self.__n)]
        else:
            self.__n: int = n  # number of evolution steps
            self.__size = int(math.log2(self.target.size) / 2)  # number of qubits

            self.phase = 0  # global phase
            self.gates = [EvolutionStep(size=self.__size) for _ in range(self.__n)]  # evolution steps

        self.evolutionGradient = [0] * self.__n  # -gradient of cost function by evolution times
        self.gradient = [[[0, 0] for _ in range(self.__size)] for _ in
                         range(self.__n)]  # -gradient of cost function by 1-qubit operations parameters
        self.stepSize = 0.01  # parameter update by gradient coefficient for 1-qubit operations parameters and
        # evolution times

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

        for i in range(self.__n):
            matrix = self.gates[i].matrix @ matrix

        return matrix * math.e ** (1j * self.phase)

    def randomize_params(self):  # randomizes params for 1-qubit operations
        for i in range(self.__n):
            self.gates[i].randomize_params()

    def calculate_gradient(self):  # grad f = (f(x + gradstep) - f(x)) / gradstep
        dist1 = self.distance

        gradstep = 0.001

        # 1-qubit operations parameters gradient
        for i in range(self.__n):
            for j in range(self.__size):
                for k in [0, 1]:
                    self.gates[i].basicGates[j].params[k] += gradstep
                    dist2 = self.distance
                    self.gates[i].basicGates[j].params[k] -= gradstep

                    self.gradient[i][j][k] = (dist1 - dist2) / gradstep * self.stepSize  # minus for descent

            # evolution times gradient
            self.gates[i].time += gradstep
            dist2 = self.distance
            self.gates[i].time -= gradstep

            self.evolutionGradient[i] = (dist1 - dist2) / gradstep * self.stepSize  # minus for descent

        # make every step + random(-1, 1) * self.random
        if self.noise:
            for i in range(self.__n):
                for j in range(self.__size):
                    for k in [0, 1]:
                        self.gradient[i][j][k] += self.noise * 2 * (random.random() - 1) * self.gradient[i][j][k]

    def correct_params(self):  # update all parameters based on gradients
        for i in range(self.__n):
            self.gates[i].correct_params(self.gradient[i], self.evolutionGradient[i])
        self.phase -= np.angle((self.matrix @ self.target_d).trace())

    @property
    def distance(self):  # Frobenius norm
        distance = ((self.matrix - self.target) @ (self.matrix - self.target).conjugate().transpose()).trace()

        return distance

    def descend(self, steps=1000, track_distance=False):  # perform gradient descent
        distances = []  # distances to track

        for i in range(steps):
            distances += [self.distance]

            self.calculate_gradient()  # calculate gradient
            self.correct_params()  # update parameters

        # most parameters are cyclic - make them in (0, max)
        for gate in self.gates:
            for basicGate in gate.basicGates:
                basicGate.params[0] = basicGate.params[0].real % (2 * math.pi)
                basicGate.params[1] = basicGate.params[1].real % (2 * math.pi)
        self.phase = self.phase.real % (2 * math.pi)

        if track_distance:
            return distances

    def to_qiskit(self):
        circuit = QuantumCircuit(self.__size, global_phase=self.phase)
        for i in range(self.__n):
            circuit += self.gates[i].to_qiskit()
        return circuit

    def to_implementation(self, name: str):
        angles = [[[gate.basicGates[i].params[0], gate.basicGates[i].params[1]] for i in range(self.__size)] for gate in self.gates]
        times = [gate.time for gate in self.gates]
        return Implementation(name, angles, times, phase=self.phase)

    def set_j(self, new_j):
        for evolution_step in self.gates:
            evolution_step.set_j(new_j)

    def make_times_positive(self): # only works for 1 and 2 qubits. Works approximately for 3 qubits
        if self.__size not in [1, 2, 3]:
            raise NotImplementedError("Making times possible only on 1, 2 and 3 qubits")
        else:
            for i in range(self.__n - 1):
                if self.gates[i].time < 0:
                    self.gates[i].time *= -1
                    print("changed")
                    if self.__size == 1:
                        self.gates[i].basicGates[0].params[0] += math.pi
                        self.gates[i + 1].basicGates[0].params[0] += math.pi
                    if self.__size == 2:
                        self.gates[i].basicGates[0].params[0] += math.pi
                        self.gates[i + 1].basicGates[0].params[0] += math.pi
                    if self.__size == 3:
                        self.gates[i].basicGates[1].params[0] += math.pi
                        self.gates[i + 1].basicGates[1].params[0] += math.pi




