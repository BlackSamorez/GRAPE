from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, eta):
        self.eta = eta
        self.t = 1

    @abstractmethod
    def update(self, w, dw):
        pass


class Identity(Optimizer):
    def __init__(self, eta=0.01):
        super().__init__(eta)

    def update(self, w, dw):
        return w - self.eta * dw


class AdamOpt(Optimizer):
    def __init__(self, size, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(eta)
        self.m_dw, self.v_dw = np.zeros((size,), dtype=float), np.zeros((size,), dtype=float)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, w, dw):
        # momentum beta 1
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw

        # rms beta 2
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)

        m_dw_corr = self.m_dw / (1 - self.beta1 ** self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** self.t)

        # update weights
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        return w
