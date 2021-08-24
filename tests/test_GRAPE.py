from unittest import TestCase
import numpy as np

from multiqubitgates import CXCascade, Pulse
from circuit import Circuit
from GRAPE import GradientOptimization

CX = np.asarray([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


class TestGradientOptimization(TestCase):
    def test_matrix(self):
        opt = GradientOptimization(CX)
        np.testing.assert_allclose(opt.matrix, CXCascade(2).matrix ** 4)

    def test_metric_distance(self):
        opt = GradientOptimization(CX)
        np.testing.assert_allclose(opt.metric_distance, 4)

    def test_metric_derivative(self):
        opt = GradientOptimization(CX)
        opt.randomize_params()
        derivative = np.random.random((4, 4))
        np.testing.assert_allclose(opt.metric_derivative(derivative),
                                   ((opt.matrix - opt.target) @ derivative.T).trace() + (
                                               (opt.matrix - opt.target).T.conjugate() @ derivative).trace())

    def test_corrections_from_gradients(self):
        circuit = Circuit(2)
        circuit += Pulse(2)
        circuit += Pulse(2)
        circuit += Pulse(2)
        circuit.gates[0].params[0] = np.pi
        opt = GradientOptimization(CX, circuit)
        opt.stepSize = 1
        opt.randomize_params()
        opt.update()
        before = opt.circuit.gates[0].params[0]
        opt.corrections_from_gradients()

        np.testing.assert_allclose(opt.circuit.gates[0].params[0], before - opt.metric_derivative(opt.circuit.derivative(0, 0)))

    def test_descend(self):
        pass
