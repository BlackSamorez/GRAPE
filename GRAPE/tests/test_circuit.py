from unittest import TestCase

import numpy as np

from GRAPE.circuit import Circuit
from GRAPE.multiqubitgates import Evolution, Pulse, CXCascade


class TestCircuit(TestCase):
    def test_create(self):
        circuit = Circuit(2)
        self.assertEqual(circuit.size, 2)
        self.assertEqual(circuit.gates, [])

    def test_iadd(self):
        circuit = Circuit(2)
        circuit += Pulse(2)
        self.assertEqual(len(circuit.gates), 1)
        self.assertEqual(type(circuit.gates[0]), Pulse)

    def test_add(self):
        circuit1 = Circuit(2)
        circuit1 += Pulse(2)
        circuit2 = Circuit(2)
        circuit2 += Pulse(2)

        add = circuit1 + circuit2
        self.assertEqual(len(add.gates), 2)
        self.assertTrue(add.gates[0] is circuit1.gates[0])
        self.assertTrue(add.gates[1] is circuit2.gates[0])

        add = add + Pulse(2)
        self.assertEqual(len(add.gates), 3)

    def test_update(self):
        circuit = Circuit(2)
        circuit += Pulse(2)
        circuit += Pulse(2)
        circuit += Pulse(2)

        circuit.randomize_params()
        circuit.update()
        np.testing.assert_allclose(circuit.gates[2].matrix @ circuit.gates[1].matrix @ circuit.gates[0].matrix,
                                   circuit.matrix)

    def test_derivative(self):
        circuit = Circuit(2)
        circuit += Pulse(2, one_qubit_gate_type="general")
        circuit += Evolution(2)
        circuit += CXCascade(2)
        circuit += Pulse(2, one_qubit_gate_type="nmr")
        circuit += Evolution(2)
        circuit += Pulse(2)

        circuit.randomize_params()
        circuit.update()

        test_increment = 0.0001
        matrix = circuit.matrix
        for i in range(len(circuit.gates)):
            for param in range(len(circuit.gates[i].params)):
                derivative = circuit.derivative(i, param)
                circuit.gates[i].params[param] += test_increment
                circuit.update()
                np.testing.assert_allclose((circuit.matrix - matrix) / test_increment, derivative,
                                           rtol=0.01)
                circuit.gates[i].params[param] -= test_increment
