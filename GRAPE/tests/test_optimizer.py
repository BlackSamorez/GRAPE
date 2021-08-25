from unittest import TestCase

import numpy as np

from GRAPE.circuit import Circuit
from GRAPE.multiqubitgates import Pulse, Evolution, CXCascade
from GRAPE.optimizer import GradientOptimization

CX = np.asarray([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


class TestGradientOptimization(TestCase):
    def test_matrix(self):
        opt = GradientOptimization(CX)
        matrix = np.eye(2 ** opt.circuit.size)
        for gate in opt.circuit.gates:
            matrix = gate.matrix @ matrix

        np.testing.assert_allclose(opt.matrix, matrix)

    def test_metric_distance(self):
        opt = GradientOptimization(CX)
        np.testing.assert_allclose(opt.metric_distance, 4)

    def test_metric_derivative(self):
        circuit = Circuit(2)
        circuit += Pulse(2, one_qubit_gate_type="general")
        circuit += Evolution(2)
        circuit += CXCascade(2)
        circuit += Pulse(2, one_qubit_gate_type="nmr")
        circuit += Evolution(2)
        circuit += Pulse(2)

        opt = GradientOptimization(CX, circuit)
        opt.randomize_params()
        opt.update()

        test_increment = 0.0001
        metric_distance = opt.metric_distance
        for gate in range(len(opt.circuit.gates)):
            for parameter in range(len(opt.circuit.gates[gate].params)):
                metric_derivative = opt.metric_derivative(opt.circuit.derivative(gate, parameter))

                opt.circuit.gates[gate].params[parameter] += test_increment
                opt.update()

                np.testing.assert_allclose((opt.metric_distance - metric_distance) / test_increment, metric_derivative,
                                           rtol=0.01)

                opt.circuit.gates[gate].params[parameter] -= test_increment
                opt.update()

    def test_corrections_from_gradients(self):
        circuit = Circuit(2)
        circuit += Pulse(2)
        circuit += Pulse(2)
        circuit += Pulse(2)
        circuit.gates[0].params[0] = np.pi
        opt = GradientOptimization(CX, circuit)
        opt.learning_rate = 1
        opt.randomize_params()
        opt.update()
        before = opt.circuit.gates[0].params[0]
        opt.corrections_from_gradients()

        np.testing.assert_allclose(opt.circuit.gates[0].params[0],
                                   before - opt.metric_derivative(opt.circuit.derivative(0, 0)))

    def test_descend(self):
        pass
