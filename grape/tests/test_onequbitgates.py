from unittest import TestCase

import numpy as np

from grape.state_vector.onequbitgates import GeneralOneQubitGate, NMROneQubitGate


class TestGeneralOneQubitGate(TestCase):
    def test_default_params(self):
        gate = GeneralOneQubitGate()
        np.testing.assert_allclose(gate.params, np.asarray([0, 0, 0], dtype=float), rtol=0.001, atol=0.001)

    def test_update_matrix(self):
        gate = GeneralOneQubitGate()
        gate.params = np.asarray([np.pi, 0, 0], dtype=float)
        gate.update_matrix()
        np.testing.assert_allclose(np.asarray([[0, -1],
                                               [1, 0]], dtype=complex), gate.matrix, rtol=0.001, atol=0.001)
        gate.params = np.asarray([0, 0, 0], dtype=float)
        gate.update_matrix()
        np.testing.assert_allclose(np.asarray([[1, 0],
                                               [0, 1]], dtype=complex), gate.matrix, rtol=0.001, atol=0.001)

    def test_randomize_params(self):
        gate = GeneralOneQubitGate()
        gate.randomize_params()
        for p in gate.params:
            self.assertEqual(type(p), np.float_)
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 2 * np.pi)

    def test_normalize(self):
        gate = GeneralOneQubitGate()
        gate.params = np.asarray([10, 15, 20], dtype=float)
        gate.update()
        matrix1 = gate.matrix
        gate.normalize()
        gate.update()
        matrix2 = gate.matrix
        np.testing.assert_allclose(matrix1, matrix2)

    def test_unitarity(self):
        gate = GeneralOneQubitGate()
        gate.randomize_params()
        gate.update()
        np.testing.assert_allclose(gate.matrix @ gate.matrix.T.conjugate(), np.eye(2, dtype=complex), rtol=0.001,
                                   atol=0.001)

    def test_derivative(self):
        gate = GeneralOneQubitGate()
        gate.randomize_params()
        gate.update()

        test_increment = 0.0001
        derivative = gate.derivative
        matrix = gate.matrix
        for param in range(len(gate.params)):
            gate.params[param] += test_increment
            gate.update()
            np.testing.assert_allclose((gate.matrix - matrix) / test_increment, derivative[param],
                                       rtol=0.01)
            gate.params[param] -= test_increment


class TestNMROneQubitGate(TestCase):
    def test_default_params(self):
        gate = NMROneQubitGate()
        np.testing.assert_allclose(gate.params, np.asarray([0, 0], dtype=float), rtol=0.001, atol=0.001)

    def test_update_matrix(self):
        gate = NMROneQubitGate()

        gate.params = np.asarray([np.pi, 0], dtype=float)
        gate.update_matrix()
        np.testing.assert_allclose(np.asarray([[0, -1j],
                                               [-1j, 0]], dtype=complex), gate.matrix, rtol=0.001, atol=0.001)

        gate.params = np.asarray([0, 0], dtype=float)
        gate.update_matrix()
        np.testing.assert_allclose(np.asarray([[1, 0],
                                               [0, 1]], dtype=complex), gate.matrix, rtol=0.001, atol=0.001)

    def test_randomize_params(self):
        gate = NMROneQubitGate()
        gate.randomize_params()
        for p in gate.params:
            self.assertEqual(type(p), np.float_)
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 2 * np.pi)

    def test_normalize(self):
        gate = NMROneQubitGate()
        gate.params = np.asarray([10, 15], dtype=float)
        gate.update()
        matrix1 = gate.matrix
        gate.normalize()
        gate.update()
        matrix2 = gate.matrix
        np.testing.assert_allclose(matrix1, matrix2)

    def test_unitarity(self):
        gate = GeneralOneQubitGate()
        gate.randomize_params()
        gate.update()
        np.testing.assert_allclose(gate.matrix @ gate.matrix.T.conjugate(), np.eye(2, dtype=complex), rtol=0.001,
                                   atol=0.001)

    def test_derivative(self):
        gate = NMROneQubitGate()
        gate.randomize_params()
        gate.update()

        test_increment = 0.00001
        derivative = gate.derivative
        matrix = gate.matrix
        for param in range(len(gate.params)):
            gate.params[param] += test_increment
            gate.update()
            np.testing.assert_allclose((gate.matrix - matrix) / test_increment, derivative[param],
                                       rtol=0.01)
            gate.params[param] -= test_increment
