import numpy as np
import math
from math import pi
import copy
import random

class basicGate():
  def __init__(self, params=[0, 0]):
    self.params = params
    self.I = np.eye(2, dtype=complex)
    self.X = np.asarray([[0, 1], [1, 0]], dtype=complex)
    self.Y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
    self.Z = np.asarray([[1, 0], [0, -1]], dtype=complex)

  @property
  def matrix(self):
    matrix = np.zeros((2, 2), dtype=np.complex)
    matrix = math.cos(self.params[0] / 2) * self.I - 1j * math.sin(self.params[0]/2) * (math.cos(self.params[1]) * self.X + math.sin(self.params[1]) * self.Y)

    return matrix

  def derivative(self, k):
    matrix = np.zeros((2, 2), dtype=np.complex)

    if k == 0:
      matrix = -math.sin(self.params[0] / 2) / 2 * self.I - 1j / 2 * math.cos(self.params[0]/2) * (math.cos(self.params[1]) * self.X + math.sin(self.params[1])* self.Y)
    if k == 1:
      matrix = -1j * math.sin(self.params[0]/2) * (-math.sin(self.params[1]) * self.X + math.cos(self.params[1]) * self.Y)
    
    return matrix

  def randomizeParams(self):
    self.params = [2 * math.pi * random.random(), 2 * math.pi * random.random()]

  def correctParams(self, correction):
    self.params[0] += correction[0]
    self.params[1] += correction[1]


class evolutionStep():
  def __init__(self, evolution = None, size = 2, time=0.01, params=[]):
    self.evoType = evolution#TEMP
    self.size = size#TODO check that is size of self.evolution
    self.time = time

    if len(params) != self.size:
      params += [[0, 0] for _ in range(self.size - len(params))]

    self.basicGates = [basicGate(param) for param in params]
      
  @property
  def evolution(self):
    if self.evoType == None and self.size == 2:
      second = -np.eye(2 ** self.size, dtype=np.complex)
      second[0][0] = 1
      second[3][3] = 1

      return (math.cos(self.time / 2) * np.eye(2 ** self.size, dtype=np.complex) - 1j * math.sin(self.time / 2) * second)
    return np.eye(2 ** self.size, dtype=np.complex)

  @property
  def evolutionDerivative(self):
    if self.evoType == None and self.size == 2:
      second = -np.eye(2 ** self.size, dtype=np.complex)
      second[0][0] = 1
      second[3][3] = 1

      derivative = (-1/2* math.sin(self.time / 2) * np.eye(2 ** self.size, dtype=np.complex) - 1j/2  * math.cos(self.time / 2) * second)

      matrix = np.ones((1), dtype=np.complex)
      for i in range(self.size):
        matrix = np.kron(matrix, self.basicGates[i].matrix)
      matrix = derivative @ matrix

      return matrix
    return np.zeroes(2 ** self.size, dtype=np.complex)
    
  @property
  def matrix(self):
    matrix = np.ones((1), dtype=np.complex)

    for i in range(self.size):
      matrix = np.kron(matrix, self.basicGates[i].matrix)
    matrix = self.evolution @ matrix

    return matrix

  def derivative(self, qubit=0, k=0):
    derivative = np.ones((1), dtype=np.complex)

    for i in range(self.size):
      if i != qubit:
        derivative = np.kron(derivative, self.basicGates[i].matrix)
      else:
        derivative = np.kron(derivative, self.basicGates[i].derivative(k))
    derivative = self.evolution @ derivative

    return derivative
    

  def randomizeParams(self):
    for basicGate in self.basicGates:
      basicGate.randomizeParams()

  def correctParams(self, correction, evolutionCorrection):      
    for i in range(self.size):
      self.basicGates[i].correctParams(correction[i])
    self.time += evolutionCorrection


class implementation():
  def __init__(self, target, n):
    self.target = target
    self.noise = 0.05

    self.n = n
    self.size = int(math.log2(self.target.size) / 2)

    self.times = [0.2] * self.n
    self.phase = 0
    self.gates = [evolutionStep(size=self.size, time=self.times[i]) for i in range(self.n)]

    self.C = []

    self.evolutionGradient = [0] * self.n
    self.phaseGradient = 0
    self.gradient = [[[0, 0] for j in range(self.size)] for i in range(self.n)]
    self.stepSize = 0.02

  @property
  def targetD(self):
    return self.target.conjugate().transpose()

  @property
  def matrix(self):
    matrix = np.eye(2**self.size, dtype=np.complex)

    for i in range(self.n):
      matrix = matrix @ self.gates[i].matrix * math.e ** (1j * self.phase)

    return matrix

  def readParams(self):
    return [gate.params for gate in self.gates]
  
  def writeParams(self, params = None):
    if params:
      for i in range(self.n):
        self.gates[i].params = params[i]
    else:
      for i in range(self.n):
        self.gates[i].randomizeParams()

  def makeC(self):
    self.C = [np.eye(2**self.size, dtype=np.complex) for _ in range(self.n)]

    for i in range(self.n):
      for k in range(i + 1, self.n):
        self.C[i] = self.gates[k].matrix @ self.C[i]
      self.C[i] = self.targetD @ self.C[i]
      for k in range(0, i):
        self.C[i] = self.gates[k].matrix @ self.C[i]

  def calculateGradient(self):
    maxGrad = 0
    for i in range(self.n):
      for j in range(self.size):
        for k in [0, 1]:
          bigDerivative = np.zeros((2**self.size, 2**self.size), dtype=np.complex)
          self.gradient[i][j][k] = (self.gates[i].derivative(j, k) @ self.C[i] * math.e ** (1j * self.phase)).trace().real * self.stepSize
          if self.gradient[i][j][k] > maxGrad:
            maxGrad = self.gradient[i][j][k]
      
    self.evolutionGradient[i] =  (self.gates[i].evolutionDerivative @ self.C[i] * math.e ** (1j * self.phase)).trace().real * self.stepSize
    
    #self.phaseGradient = - (self.matrix).trace().imag * self.stepSize

    #if self.noise:
    #  for i in range(self.n):
    #    for j in range(self.size):
    #      for k in [0, 1]:
    #       self.gradient[i][j][k] += self.noise * 2 *(random.random() - 1) * self.gradient[i][j][k]
      

  def correctParams(self):
    for i in range(self.n):
      self.gates[i].correctParams(self.gradient[i], self.evolutionGradient[i])

  @property
  def distance(self):
    distance = ((self.matrix - self.target) @ (self.matrix - self.target).conjugate().transpose()).trace()

    return distance

  def descend(self, steps=1000, trackDistance=False):
    distances = []
    
    for i in range(steps):
      distances += [self.distance]
      self.makeC()
      self.calculateGradient()
      self.correctParams()
    if trackDistance:
      return distances
