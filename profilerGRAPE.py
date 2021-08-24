import matplotlib.pyplot as plt
import numpy as np
from multiqubitgates import Pulse, Delay, Inversion, CXCascade
from circuit import Circuit
from GRAPE import GradientOptimization

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
circuit += Delay(3)
circuit += Pulse(3, "nmr")
circuit += Delay(3)
circuit += Pulse(3, "nmr")
circuit += Delay(3)
circuit += Pulse(3, "nmr")


desc = GradientOptimization(toffoli, circuit)
desc.randomize_params()
desc.descend(steps=1000)
