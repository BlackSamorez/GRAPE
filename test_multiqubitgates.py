from unittest import TestCase
import numpy as np

from multiqubitgates import Delay, Pulse, Inversion, CXCascade


def test_params(gate, test_entry):
    for i in range(len(gate.params)):
        old = gate.params[i]
        gate.params[i] = gate.params[i]
        new = gate.params[i]
        test_entry.assertEqual(old, new)


class TestPulse(TestCase):
    def test_default_values(self):
        pulse = Pulse(2)
        self.assertEqual(pulse.time, 0)
        self.assertEqual(pulse.size, 2)
        self.assertEqual(pulse.derivative.shape, (2 * 3, 4, 4))
        self.assertEqual(len(pulse.basic_gates), 2)

    def test_update_matrix(self):
        pulse = Pulse(2)
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

    def test_to_circuit(self):
        pulse = Pulse(2)
        pulse.randomize_params()
        pulse.update()
        circuit = pulse.to_circuit()
        assert circuit.size() == pulse.size

    def test_unitarity(self):
        pulse = Pulse(2)
        pulse.randomize_params()
        pulse.update()

        np.testing.assert_allclose(pulse.matrix @ pulse.matrix.T.conjugate(), np.eye(4, dtype=complex), rtol=0.001,
                                   atol=0.001)

    def test_params(self):
        pulse = Pulse(2)
        self.assertEqual(len(pulse.params), pulse.basic_gates[0].number_of_parameters * pulse.size)
        test_params(pulse, self)


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

    def test_to_circuit(self):
        pulse = Pulse(2)
        pulse.randomize_params()
        pulse.update()
        circuit = pulse.to_circuit()
        assert circuit.size() == pulse.size

    def test_params(self):
        inversion = Inversion(2)
        test_params(inversion, self)


class TestCXCascade(TestCase):
    def test_creation(self):
        cx_cascade = CXCascade(2)

    def test_unitarity(self):
        cx_cascade = CXCascade(2)
        np.testing.assert_allclose(cx_cascade.matrix @ cx_cascade.matrix.T.conjugate(), np.eye(4, dtype=complex),
                                   rtol=0.001, atol=0.001)

    def test_entanglement(self):
        pass

    def test_params(self):
        cx_cascade = Inversion(2)
        test_params(cx_cascade, self)


class TestDelay(TestCase):
    def test_params(self):
        delay = Delay(2)
        test_params(delay, self)
