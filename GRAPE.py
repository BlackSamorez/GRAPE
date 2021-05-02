import numpy as np
import math
import random
from qiskit import QuantumCircuit


class BasicGate:  # 1-qubit gate
    def __init__(self, params=None):
        if params is None:
            params = [0, 0]
        self.params = params
        # Pauli matrices
        self.I = np.eye(2, dtype=complex)
        self.X = np.asarray([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)

    @property
    def matrix(self):  # straightforward matrix representation
        matrix = math.cos(self.params[0] / 2) * self.I - 1j * math.sin(self.params[0] / 2) * (
                    math.cos(self.params[1]) * self.X + math.sin(self.params[1]) * self.Y)

        return matrix

    def randomize_params(self):
        self.params = [2 * math.pi * random.random(), 2 * math.pi * random.random()]

    def correct_params(self, correction):  # update parameters based on passed corrections
        self.params[0] += correction[0]
        self.params[1] += correction[1]


class EvolutionStep:  # a combination of 1-qubit operations followed by evolution
    def __init__(self, size=2, params=None):
        if params is None:
            params = []
        self.size = size  # number of qubits
        self.time = 0.2  # evolution time

        if len(params) != self.size:
            params += [[0, 0] for _ in range(self.size - len(params))]

        self.basicGates = [BasicGate(param) for param in params]  # 1-qubit gates

        self.J = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)  # interaction matrix (NOT FINISHED)
        for i in range(self.size - 1):
            self.J[i + 1][i] = 1  # костыль

        self.Z = np.asarray([[1, 0], [0, -1]], dtype=complex)
        self.sigmas = [np.ones(1) for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    self.sigmas[j] = np.kron(self.sigmas[j], self.Z)
                else:
                    self.sigmas[j] = np.kron(self.sigmas[j], np.eye(2))

    @property
    def hamiltonian(self):
        hamiltonian = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)
        for i in range(self.size):
            for j in range(i):
                hamiltonian += math.pi / 2 * self.J[i][j] * self.sigmas[i] @ self.sigmas[j]
        return hamiltonian

    @property
    def evolution(self):  # evolution matrix
        evolution = np.eye(2 ** self.size, dtype=complex)
        for i in range(self.size):
            for j in range(i):
                evolution = evolution @ (math.cos(math.pi / 2 * self.J[i][j] * self.time) * np.eye(2 ** self.size,
                                                                                                   dtype=complex) - 1j * math.sin(
                    math.pi / 2 * self.J[i][j] * self.time) * self.sigmas[i] @ self.sigmas[j])
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


class Implementation:  # class to approximate abstract unitary using a series of evolutionStep
    def __init__(self, target, n):
        self.target = target  # unitary to approximate
        self.noise = 0.05  # noise levels (see self.stupidCalculateGradient)

        self.n = n  # number of evolution steps
        self.size = int(math.log2(self.target.size) / 2)  # number of qubits

        self.phase = 0  # global phase
        self.gates = [EvolutionStep(size=self.size) for _ in range(self.n)]  # evolution steps

        self.evolutionGradient = [0] * self.n  # -gradient of cost function by evolution times
        self.phaseGradient = 0  # -gradient of cost function by global phase
        self.gradient = [[[0, 0] for _ in range(self.size)] for _ in
                         range(self.n)]  # -gradient of cost function by 1-qubit operations parameters
        self.stepSize = 0.01  # parameter update by gradient coefficient for 1-qubit operations parameters and
        # evolution times
        self.phaseStepSize = 0.002  # parameter update by gradient coefficient for global phase (diverges when trying
        # to lower)

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
        matrix = np.eye(2 ** self.size, dtype=complex)

        for i in range(self.n):
            matrix = self.gates[i].matrix @ matrix

        return matrix * math.e ** (1j * self.phase)

    def write_params(self, params=None):  # writes params for 1-qubit operations
        if params:
            for i in range(self.n):
                self.gates[i].params = params[i]
        else:
            for i in range(self.n):
                self.gates[i].randomize_params()  # randomizes if no input

    def save_params(self, filename):
        # self.gates[0].basicGates[0].params[0], self.gates[0].basicGates[0].params[1]; ... ; self.gates[0].basicGates[self.size - 1].params[0]; self.gates[0].basicGates[self.size - 1].params[1]; self.gates[0].time
        # ...
        # self.gates[self.n - 1].basicGates[0].params[0], self.gates[0].basicGates[0].params[1]; ... ; self.gates[0].basicGates[self.size - 1].params[0]; self.gates[self.n - 1].basicGates[self.size - 1].params[1]; self.gates[self.n - 1].time
        # self.phase; ... ; self.phase

        file = open(filename, "w")
        for i in range(self.n):
            for j in range(self.size):
                file.write(f"{self.gates[i].basicGates[j].params[0]}; {self.gates[i].basicGates[j].params[1]}; ")
            file.write(f"{self.gates[i].time}\n")
        for i in range(2 * self.size + 1):
            file.write(f"{self.phase}")
            if i != 2 * self.size:
                file.write("; ")
        file.write("\n")

    def read_params(self, filename):
        file = open(filename, "r")
        data = np.genfromtxt(filename, delimiter=";")
        for i in range(self.n):
            for j in range(self.size):
                self.gates[i].basicGates[j].params[0] = data[i][2 * j]
                self.gates[i].basicGates[j].params[1] = data[i][2 * j + 1]
            self.gates[i].time = data[i][2 * self.size]
        self.phase = data[self.n][0]

    def calculate_gradient(self):  # grad f = (f(x + gradstep) - f(x)) / gradstep
        dist1 = self.distance

        gradstep = 0.001

        # 1-qubit operations parameters gradient
        for i in range(self.n):
            for j in range(self.size):
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

        # global phase gradient
        self.phase += gradstep
        dist2 = self.distance
        self.phase -= gradstep

        self.phaseGradient += (dist1 - dist2) / gradstep * self.phaseStepSize  # minus for descent

        # make every step + random(-1, 1) * self.random
        if self.noise:
            for i in range(self.n):
                for j in range(self.size):
                    for k in [0, 1]:
                        self.gradient[i][j][k] += self.noise * 2 * (random.random() - 1) * self.gradient[i][j][k]

    def correct_params(self):  # update all parameters based on gradients
        for i in range(self.n):
            self.gates[i].correct_params(self.gradient[i], self.evolutionGradient[i])
        self.phase = self.phaseGradient

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
        # print("prior", self.distance)
        for gate in self.gates:
            for basicGate in gate.basicGates:
                basicGate.params[0] = basicGate.params[0].real % (4 * math.pi)
                basicGate.params[1] = basicGate.params[1].real % (2 * math.pi)
            # gate.time = gate.time.real % (4 * math.pi)
        self.phase = self.phase.real % (2 * math.pi)
        # print("after", self.distance)

        if track_distance:
            return distances

    def to_qiskit(self):
        circuit = QuantumCircuit(self.size, global_phase=self.phase)
        for i in range(self.n):
            circuit += self.gates[i].to_qiskit()
        return circuit
