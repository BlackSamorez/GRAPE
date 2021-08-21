import numpy as np
import math
from math import pi
import random
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod

from multiqubitgates import MultiQubitGate, Delay, Pulse, Inversion, CXCascade
from circuit import Circuit, OneQubitEntanglementAlternation


class GradientOptimization:
    def __init__(self, target, architecture: Circuit = None, filename: str = None, entanglement_gate_type: type = CXCascade):
        self.target = target  # unitary to approximate

        if filename is not None:
            self.read_text(filename)
        else:
            self._size = int(math.log2(self.target.size) / 2)  # number of qubits
            self.phase = 0  # global phase

            if architecture is None:
                architecture = OneQubitEntanglementAlternation(entanglement_gate_type, 2 * self._size)
            self.gates: list[MultiQubitGate] = []  # simultaneous gates
            for i in range(len(architecture)):
                self.gates.append(architecture.gates[i](self._size))
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
        return ((self.matrix - self.target) @ (self.matrix - self.target).conjugate().transpose()).trace().real

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
                # noinspection PyArgumentList
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
                                    self.matrix - self.target).conjugate().transpose()).trace() + self.distance / self.approx_time).real * math.e ** (
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
                                        matrix @ (self.matrix - self.target).conjugate().transpose()).trace()).real

    def descend(self, steps=1000, track_distance=False, time_sensitive=False):
        distances = []  # distances to track

        for i in range(steps):
            distances.append(self.distance)
            self.corrections_from_gradients(time_sensitive=time_sensitive)
            self.update()
            self.phase -= np.angle((self.matrix @ self.target_d).trace())

        # most parameters are cyclic - make them in (0, max)
        for gate in self.gates:
            gate.normalize()

        self.phase = self.phase.real % (2 * math.pi)

        if track_distance:
            return distances

    def to_qiskit(self):
        circuit = QuantumCircuit(self._size, global_phase=self.phase)
        for gate in self.gates:
            circuit += gate.to_circuit()
        return circuit

    def set_j(self, new_j):
        for gate in self.gates:
            if type(gate) is Delay:
                gate.set_j(new_j)

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
                self.gates.append(Delay(size=self._size, time=float(data[1])))
            if data[0] == "pulse" or data[0] == "Kick":
                params = []
                for j in range(self._size):
                    params.append([float(data[2 * j + 1]), float(data[2 * j + 2])])
                self.gates.append(Pulse(size=self._size, params=params))
            if data[0] == "inversion" or data[0] == "Inversion":
                self.gates.append(Inversion(size=self._size, qubits=[int(qubit) for qubit in data[1:]]))
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
