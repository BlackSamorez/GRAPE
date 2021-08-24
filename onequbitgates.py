import numpy as np
import random
from abc import ABC, abstractmethod


class OneQubitGate(ABC):
    """Abstract Base Class for one qubit gates"""

    def __init__(self, number_of_parameters: int, params: np.ndarray = None):
        self._id = np.eye(2, dtype=complex)
        self._x = np.asarray([[0, 1],
                              [1, 0]], dtype=complex)
        self._y = np.asarray([[0, -1j],
                              [1j, 0]], dtype=complex)

        self.matrix: np.ndarray = np.ones(2, dtype=complex)
        self.number_of_parameters = number_of_parameters
        self.derivative: np.ndarray = np.zeros((self.number_of_parameters, 2, 2), dtype=complex)
        if params is None:
            params = np.zeros(self.number_of_parameters, dtype=float)
        self.params: np.ndarray = np.asarray(params, dtype=float)
        assert self.params.shape == (self.number_of_parameters,), f"Mismatch in number_of_params and shape of params provided: {self.number_of_parameters} vs {self.params.shape[0]} "

    @abstractmethod
    def update_matrix(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def randomize_params(self):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class GeneralOneQubitGate(OneQubitGate):
    """Representation of a general one qubit gate"""
    def __init__(self, params=None):
        super().__init__(3, params=params)  # params: theta, phi, lambda

    def update_matrix(self):
        self.matrix = np.asarray(
            [[np.cos(self.params[0] / 2), -np.e ** (1j * self.params[2]) * np.sin(self.params[0] / 2)],
             [np.e ** (1j * self.params[1]) * np.sin(self.params[0] / 2),
              np.e ** (1j * (self.params[1] + self.params[2])) * np.cos(self.params[0] / 2)]], dtype=complex)

    def update_derivative(self):
        d_theta = np.asarray(
            [[-np.sin(self.params[0] / 2) / 2, -np.e ** (1j * self.params[2]) * np.cos(self.params[0] / 2) / 2],
             [np.e ** (1j * self.params[1]) * np.cos(self.params[0] / 2) / 2,
              -np.e ** (1j * (self.params[1] + self.params[2])) * np.sin(self.params[0] / 2) / 2]], dtype=complex)
        d_phi = np.asarray([[0, 0],
                            [1j * np.e ** (1j * self.params[1]) * np.sin(self.params[0] / 2),
                             1j * np.e ** (1j * (self.params[1] + self.params[2])) * np.cos(self.params[0] / 2)]],
                           dtype=complex)
        d_lambda = np.asarray(
            [[0, -1j * np.e ** (1j * self.params[2]) * np.sin(self.params[0] / 2)],
             [0, 1j * np.e ** (1j * (self.params[1] + self.params[2])) * np.cos(self.params[0] / 2)]], dtype=complex)
        self.derivative[0] = d_theta
        self.derivative[1] = d_phi
        self.derivative[2] = d_lambda

    def update(self):
        self.update_matrix()
        self.update_derivative()

    def randomize_params(self):
        self.params = np.asarray([2 * np.pi * random.random(), 2 * np.pi * random.random(), 2 * np.pi * random.random()], dtype=float)

    def normalize(self):
        self.params[0] = self.params[0] % (4 * np.pi)
        self.params[1] = self.params[1] % (2 * np.pi)
        self.params[2] = self.params[2] % (2 * np.pi)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.params}"


class NMROneQubitGate(OneQubitGate):
    """One qubit gate limited to rotations in XY plane"""
    def __init__(self, params: np.ndarray = None):
        super().__init__(2, params)  # params: theta, phi

    def update_matrix(self):  # straightforward matrix representation
        self.matrix = np.cos(self.params[0] / 2) * self._id - 1j * np.sin(self.params[0] / 2) * (
                np.cos(self.params[1]) * self._x + np.sin(self.params[1]) * self._y)

    def update_derivative(self):
        d_theta = -1 / 2 * np.sin(self.params[0] / 2) * self._id - 1j / 2 * np.cos(self.params[0] / 2) * (
                np.cos(self.params[1]) * self._x + np.sin(self.params[1]) * self._y)
        d_phi = 1j * np.sin(self.params[0] / 2) * (
                np.sin(self.params[1]) * self._x + np.cos(self.params[1]) * self._y)
        self.derivative[0] = d_theta
        self.derivative[1] = d_phi

    def update(self):
        self.update_matrix()
        self.update_derivative()

    def randomize_params(self):
        self.params = np.asarray([2 * np.pi * random.random(), 2 * np.pi * random.random()])

    def normalize(self):  # no update needed
        self.params[0] = self.params[0].real % (4 * np.pi)
        self.params[1] = self.params[1].real % (2 * np.pi)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.params}"
