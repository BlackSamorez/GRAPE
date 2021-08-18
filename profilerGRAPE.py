import numpy as np
from GRAPE import GradientDescent

J = np.zeros((3, 3))
J[0][1] = 0.1385
J[1][2] = 0.01304
J[0][2] = 0.00148

TOFFOLI = np.eye(8)
TOFFOLI[6][6] = 0
TOFFOLI[7][7] = 0
TOFFOLI[6][7] = 1
TOFFOLI[7][6] = 1

desc = GradientDescent(TOFFOLI, n=6)
desc.set_j(J)
desc.randomize_params()
desc.descend(time_sensitive=True)