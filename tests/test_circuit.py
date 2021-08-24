from unittest import TestCase
import numpy as np
from circuit import Circuit
from multiqubitgates import Pulse


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
        circuit += Pulse(2)
        circuit += Pulse(2)
        circuit += Pulse(2)

        circuit.randomize_params()
        circuit.update()
        for parameter in range(len(circuit.gates[2].params)):
            np.testing.assert_allclose(
                circuit.gates[2].derivative[parameter] @ circuit.gates[1].matrix @ circuit.gates[0].matrix,
                circuit.derivative(2, parameter))
