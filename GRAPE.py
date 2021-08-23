import numpy as np
import math
from math import pi
import random
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod

from multiqubitgates import MultiQubitGate, Delay, Pulse, Inversion, CXCascade
from circuit import Circuit, OneQubitEntanglementAlternation


class GradientOptimization:
    def __init__(self, target, circuit: Circuit = None, filename: str = None):
        self.target = target  # unitary to approximate
        self.stepSize = 0.01  # gradient-to-change ration
        self.noise = 0
        self.time_weight_constant = 300

        if filename is not None:
            raise NotImplementedError
            # self.read_text(filename)
        else:
            self._size = int(math.log2(self.target.size) / 2)  # number of qubits
            self.phase = 0  # global phase

            if circuit is None:
                self.circuit = OneQubitEntanglementAlternation(entanglement_gate_type, 2 * self._size)
            else:
                self.circuit = circuit
            self.gates: list[MultiQubitGate] = []  # simultaneous gates

        self.matrix = np.ones((2 ** self._size, 2 ** self._size), dtype=complex)
        self.update()

    @property
    def time(self) -> float:
        """
        Total runtime of contained circuit

        :return: total runtime of contained circuit
        """
        return self.circuit.time

    @property
    def target_d(self):
        """Target^dagger"""
        return self.target.conjugate().transpose()

    @property
    def metric_distance(self):
        """Frobenius norm between matrix and target"""
        return ((self.circuit.matrix * np.e ** (1j * self.phase) - self.target) @ (self.circuit.matrix * np.e ** (1j * self.phase) - self.target).conjugate().transpose()).trace().real

    def update(self):
        """Update the circuit"""
        self.circuit.update()

    def randomize_params(self):
        """Randomize circuit params"""
        self.circuit.randomize_params()

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
                                    self.matrix - self.target).conjugate().transpose()).trace() + self.metric_distance / self.time_weight_constant).real * math.e ** (
                                                      self.time / self.time_weight_constant)
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
            distances.append(self.metric_distance)
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

    def print_times(self):
        for gate in self.circuit.gates:
            print(gate.time, end=" ")
        print("\n")
