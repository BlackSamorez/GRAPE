import numpy as np

from grape.state_vector.circuit import Circuit, OneQubitEntanglementAlternation
from grape.state_vector.multiqubitgates import Evolution
from grape.optimizer import Optimizer, AdamOpt, Identity


class GradientOptimization:
    def __init__(self, target, circuit: Circuit = None, optimizer: str = None, filename: str = None):
        self.target = target  # unitary to approximate

        if filename is not None:
            raise NotImplementedError
            # self.read_text(filename)
        else:
            self._size = int(np.log2(self.target.size) / 2)  # number of qubits
            self.phase = 0  # global phase

            if circuit is None:
                self.circuit = OneQubitEntanglementAlternation(self._size, entanglement_gate_type=Evolution,
                                                               number_of_entanglements=2 * self._size)
            else:
                self.circuit = circuit
        self.set_hamiltonian = self.circuit.set_hamiltonian
        self.update()

        if optimizer is None:
            self.optimizer = Identity()
        elif optimizer == "adam":
            self.optimizer = AdamOpt(len(self.circuit.params))
        else:
            raise ValueError(f'optimizer must be None or "adam ", {optimizer} was given')

    @property
    def matrix(self):
        return self.circuit.matrix * np.e ** (1j * self.phase)

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
        return self.target.conjugate().T

    @property
    def metric_distance(self):
        """Frobenius norm between matrix and target"""
        return ((self.matrix - self.target) @ (
                self.matrix - self.target).conjugate().T).trace().real

    def metric_derivative(self, derivative):
        """
        Frobenius norm derivative calculation

        :param derivative: derivative of matrix
        :return: derivative of norm
        """
        return (((self.matrix - self.target) @ derivative.conjugate().T).trace() + (
                derivative @ (self.matrix - self.target).conjugate().T).trace()).real

    def update_phase(self):
        """Set optimal phase"""
        self.phase += -np.angle((self.matrix @ self.target.T).trace())

    def update(self):
        """Update the circuit"""
        self.circuit.update()
        self.update_phase()

    def randomize_params(self):
        """Randomize circuit params"""
        self.circuit.randomize_params()

    def corrections_from_gradients(self):
        derivatives = np.zeros((len(self.circuit.params)), dtype=float)
        for i in range(len(self.circuit.params)):
            derivatives[i] = self.metric_derivative(self.circuit.derivative(i))
        self.circuit.params = self.optimizer.update(self.circuit.params, derivatives)

    def descend(self, steps=1000, track_distance=False):
        distances = []  # distances to track

        self.update()
        for i in range(steps):
            distances.append(self.metric_distance)
            self.corrections_from_gradients()
            self.update()
            # self.phase = np.angle((self.target_d @ self.matrix).trace())

        # most parameters are cyclic - make them in (0, max)
        self.circuit.normalize()
        self.update()

        if track_distance:
            return distances

    def make_times_positive(self):
        raise NotImplementedError
        # if self._size not in [1, 2, 3]:
        #     raise NotImplementedError("Making time positive only possible for 1, 2 and 3 qubit systems")
        # all_positive = False
        # while not all_positive:
        #     for i in range(len(self.gates)):
        #         if type(self.gates[i]) is Delay and self.gates[i].time < 0:
        #             if type(self.gates[i + 1]) is Inversion:
        #                 # print(f"popped inversion at {i}")
        #                 self.gates[i].time *= -1
        #                 self.gates.pop(i + 1)
        #                 self.gates.pop(i - 1)
        #                 break
        #             if type(self.gates[i + 1]) is Pulse:
        #                 # print(f"added inversion at {i}")
        #                 self.gates[i].time *= -1
        #                 self.gates.insert(i + 1, Inversion(self._size, [1]))
        #                 self.gates.insert(i, Inversion(self._size, [1]))
        #                 break
        #         if i == len(self.gates) - 1:
        #             all_positive = True

    def print_times(self):
        for gate in self.circuit.gates:
            print(gate.time, end=" ")
        print("\n")

    def to_qiskit(self):
        from qiskit import QuantumCircuit
        circuit = self.circuit.to_qiskit()
        circuit.global_phase = self.phase
        return circuit
