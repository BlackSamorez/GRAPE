import numpy as np

from grape.state_vector.circuit import Circuit
from grape.state_vector.multiqubitgates import Pulse, Evolution
from grape.state_vector.gradient_optimization import GradientOptimization

cx = np.asarray([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

toffoli = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)

circuit = Circuit(3)
circuit += Pulse(3, "nmr")
circuit += Evolution(3)
circuit += Pulse(3, "nmr")
circuit += Evolution(3)
circuit += Pulse(3, "nmr")
circuit += Evolution(3)
circuit += Pulse(3, "nmr")

desc = GradientOptimization(toffoli, circuit)
desc.randomize_params()
desc.descend(steps=1000)
