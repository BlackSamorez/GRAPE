from unittest import TestCase

import numpy as np

from grape.state_vector.multiqubitgates import Evolution, Pulse, Inversion, CXCascade
from grape.state_vector.onequbitgates import GeneralOneQubitGate


class TestPulse(TestCase):
    def test_default_values(self):
        pulse = Pulse(2)
        self.assertEqual(pulse.time, 0)
        self.assertEqual(pulse.size, 2)
        self.assertEqual(pulse.derivative.shape, (2 * pulse.basic_gates[0].number_of_parameters, 4, 4))
        self.assertEqual(len(pulse.basic_gates), 2)

    def test_update_matrix(self):
        pulse = Pulse(2, one_qubit_gate_type=GeneralOneQubitGate)
        np.testing.assert_allclose(pulse.matrix, np.eye(4, dtype=complex), rtol=0.001, atol=0.001)
        pulse.basic_gates[0].params[0] = np.pi
        pulse.basic_gates[0].update()
        pulse.update_matrix()
        np.testing.assert_allclose(pulse.matrix, np.asarray([[0, -1, 0, 0],
                                                             [1, 0, 0, 0],
                                                             [0, 0, 0, -1],
                                                             [0, 0, 1, 0]], dtype=complex), rtol=0.001, atol=0.001)

    def test_update_derivative(self):
        pulse = Pulse(2)
        for i in range(pulse.basic_gates[0].number_of_parameters):
            np.testing.assert_allclose(pulse.derivative[i],
                                       np.kron(pulse.basic_gates[1].matrix, pulse.basic_gates[0].derivative[i]),
                                       rtol=0.001, atol=0.001)
        for i in range(pulse.basic_gates[0].number_of_parameters):
            np.testing.assert_allclose(pulse.derivative[pulse.basic_gates[0].number_of_parameters + i],
                                       np.kron(pulse.basic_gates[1].derivative[i], pulse.basic_gates[0].matrix),
                                       rtol=0.001, atol=0.001)

    def test_derivative(self):
        pulse = Pulse(2)
        pulse.randomize_params()
        pulse.update()

        test_increment = 0.0001
        derivative = pulse.derivative
        matrix = pulse.matrix
        for param in range(len(pulse.params)):
            pulse.params[param] += test_increment
            pulse.update()
            np.testing.assert_allclose((pulse.matrix - matrix) / test_increment, derivative[param],
                                       rtol=0.01)
            pulse.params[param] -= test_increment

    def test_derivative_nmr(self):
        pulse = Pulse(2, one_qubit_gate_type="nmr")
        pulse.randomize_params()
        pulse.update()

        test_increment = 0.0001
        derivative = pulse.derivative
        matrix = pulse.matrix
        for param in range(len(pulse.params)):
            pulse.params[param] += test_increment
            pulse.update()
            np.testing.assert_allclose((pulse.matrix - matrix) / test_increment, derivative[param],
                                       rtol=0.01)
            pulse.params[param] -= test_increment

    def test_unitarity(self):
        pulse = Pulse(2)
        pulse.randomize_params()
        pulse.update()

        np.testing.assert_allclose(pulse.matrix @ pulse.matrix.T.conjugate(), np.eye(4, dtype=complex), rtol=0.001,
                                   atol=0.001)

    def test_params(self):
        pulse = Pulse(2)
        self.assertEqual(len(pulse.params), pulse.basic_gates[0].number_of_parameters * pulse.size)
        for i in range(len(pulse.params)):
            old = pulse.params[i]
            pulse.params[i] = pulse.params[i]
            new = pulse.params[i]
            self.assertEqual(old, new)


class TestInversion(TestCase):
    def test_default_values(self):
        inversion = Inversion(2)
        self.assertEqual(inversion.qubits, [])

    def test_unitarity(self):
        inversion = Inversion(2, qubits=[0])
        np.testing.assert_allclose(inversion.matrix @ inversion.matrix.T.conjugate(), np.eye(4, dtype=complex),
                                   rtol=0.001, atol=0.001)

    def test_double_application(self):
        inversion = Inversion(2, qubits=[0])
        np.testing.assert_allclose(inversion.matrix @ inversion.matrix, np.eye(4, dtype=complex), rtol=0.001,
                                   atol=0.001)

    def test_qubit_order(self):
        inversion = Inversion(2, qubits=[0])
        np.testing.assert_allclose(inversion.matrix, np.asarray([[0, 1, 0, 0],
                                                                 [1, 0, 0, 0],
                                                                 [0, 0, 0, 1],
                                                                 [0, 0, 1, 0]], dtype=complex), rtol=0.001, atol=0.001)


class TestCXCascade(TestCase):
    def test_creation(self):
        cx_cascade = CXCascade(2)

    def test_unitarity(self):
        cx_cascade = CXCascade(2)
        np.testing.assert_allclose(cx_cascade.matrix @ cx_cascade.matrix.T.conjugate(), np.eye(4, dtype=complex),
                                   rtol=0.001, atol=0.001)

    def test_entanglement(self):
        pass


class TestDelay(TestCase):
    def test_params(self):
        delay = Evolution(2)
        for i in range(len(delay.params)):
            old = delay.params[i]
            delay.params[i] = delay.params[i]
            new = delay.params[i]
            self.assertEqual(old, new)

    def test_derivative(self):
        delay = Evolution(2)
        delay.randomize_params()
        delay.update()

        test_increment = 0.0001
        derivative = delay.derivative
        matrix = delay.matrix
        for param in range(len(delay.params)):
            delay.params[param] += test_increment
            delay.update()
            np.testing.assert_allclose((delay.matrix - matrix) / test_increment, derivative[param],
                                       rtol=0.01)
            delay.params[param] -= test_increment
