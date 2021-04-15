import numpy as np
import math
from math import pi
import copy
import random

class basicGate(): # 1-qubit gate
  def __init__(self, params=[0, 0]):
    self.params = params
    # Pauli matrices
    self.I = np.eye(2, dtype=complex) 
    self.X = np.asarray([[0, 1], [1, 0]], dtype=complex)
    self.Y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
    self.Z = np.asarray([[1, 0], [0, -1]], dtype=complex)

  @property
  def matrix(self): # straightforward matrix representation
    matrix = np.zeros((2, 2), dtype=np.complex)
    matrix = math.cos(self.params[0] / 2) * self.I - 1j * math.sin(self.params[0]/2) * (math.cos(self.params[1]) * self.X + math.sin(self.params[1]) * self.Y)

    return matrix

  def randomizeParams(self):
    self.params = [2 * math.pi * random.random(), 2 * math.pi * random.random()]

  def correctParams(self, correction): # update parameters based on passed corrections
    self.params[0] += correction[0]
    self.params[1] += correction[1]


class evolutionStep(): # a combination of 1-qubit operations followed by evolution
  def __init__(self, size = 2, params=[]):
    self.size = size # number of qubits
    self.time = 0.2 # evolution time

    if len(params) != self.size:
      params += [[0, 0] for _ in range(self.size - len(params))]

    self.basicGates = [basicGate(param) for param in params] # 1-qubit gates

    self.J = np.zeros((2 ** self.size, 2 ** self.size), dtype=np.complex) # interaction matrix (NOT FINISHED)
    for i in range(self.size - 1):
      self.J[i + 1][i] = 1 # костыль
      
  @property
  def evolution(self): # evolution matrix
    if self.size == 2:
      second = -np.eye(2 ** self.size, dtype=np.complex)
      second[0][0] = 1
      second[3][3] = 1

      return (math.cos(self.time / 2) * np.eye(2 ** self.size, dtype=np.complex) - 1j * math.sin(self.time / 2) * second)
    return np.eye(2 ** self.size, dtype=np.complex) # NOT FINISHED
    
  @property
  def matrix(self): # unitary of this evolution step
    matrix = np.ones((1), dtype=np.complex)

    for i in range(self.size):
      matrix = np.kron(matrix, self.basicGates[i].matrix) # 1-qubit "kick"
    matrix = self.evolution @ matrix # evolution

    return matrix

  def randomizeParams(self):
    for basicGate in self.basicGates:
      basicGate.randomizeParams()

  def correctParams(self, correction, evolutionCorrection): # update parameters based on passed corrections     
    for i in range(self.size):
      self.basicGates[i].correctParams(correction[i])
    self.time += evolutionCorrection


class implementation(): # class to approximate abstract unitary using a series of evolutionStep
  def __init__(self, target, n):
    self.target = target # unitary to approximate
    self.noise = 0.05 # noise levels (see self.stupidCalculateGradient)

    self.n = n # number of evolution steps
    self.size = int(math.log2(self.target.size) / 2) # number of qubits

    self.phase = 0 # global phase
    self.gates = [evolutionStep(size=self.size) for i in range(self.n)] # evolution steps

    self.evolutionGradient = [0] * self.n # -gradient of cost function by evolution times
    self.phaseGradient = 0 # -gradient of cost function by global phase
    self.gradient = [[[0, 0] for j in range(self.size)] for i in range(self.n)] # -gradient of cost function by 1-qubit operations parameters
    self.stepSize = 0.05 # parameter update by gradient coefficient for 1-qubit operations parameters and evolution times
    self.phaseStepSize = 0.01 # parameter update by gradient coefficient for global phase (diverges when trying to lower)

  @property
  def time(self): # total approximation time
    time = 0
    for gate in self.gates:
      time += gate.time
    return time.real

  @property
  def targetD(self): # target dagger
    return self.target.conjugate().transpose()

  @property
  def matrix(self): # approximation matrix
    matrix = np.eye(2**self.size, dtype=np.complex)

    for i in range(self.n):
      matrix = matrix @ self.gates[i].matrix * math.e ** (1j * self.phase)

    return matrix
  
  def writeParams(self, params = None): # writes params for 1-qubit operations
    if params:
      for i in range(self.n):
        self.gates[i].params = params[i]
    else:
      for i in range(self.n):
        self.gates[i].randomizeParams() # randomizes if no input

  def stupidCalculateGradient(self): # grad f = (f(x + gradstep) - f(x)) / gradstep
    dist1 = self.distance

    gradstep = 0.001

    # 1-qubit operations parameters gradient
    for i in range(self.n):
      for j in range(self.size):
        for k in [0, 1]:
          self.gates[i].basicGates[j].params[k] += gradstep
          dist2 = self.distance
          self.gates[i].basicGates[j].params[k] -= gradstep

          self.gradient[i][j][k] = (dist1 - dist2) / gradstep * self.stepSize # minus for descent

      # evolution times gradient
      self.gates[i].time += gradstep
      dist2 = self.distance
      self.gates[i].time -= gradstep
      
      self.evolutionGradient[i] = (dist1 - dist2) / gradstep * self.stepSize # minus for descent

    # global phase gradient
    self.phase += gradstep
    dist2 = self.distance
    self.phase -= gradstep

    self.phaseGradient += (dist1 - dist2) / gradstep * self.phaseStepSize # minus for descent
    
    # make every step + random(-1, 1) * self.random
    if self.noise:
      for i in range(self.n):
        for j in range(self.size):
          for k in [0, 1]:
           self.gradient[i][j][k] += self.noise * 2 *(random.random() - 1) * self.gradient[i][j][k]
  
  def correctParams(self): # update all parameters based on gradients
    for i in range(self.n):
      self.gates[i].correctParams(self.gradient[i], self.evolutionGradient[i])
    self.phase = self.phaseGradient
  
  @property
  def distance(self): # Frobenius norm
    distance = ((self.matrix - self.target) @ (self.matrix - self.target).conjugate().transpose()).trace()

    return distance

  def descend(self, steps=1000, trackDistance=False, stupid=True): # perform gradient descent
    distances = [] # distances to track
    
    for i in range(steps):
      distances += [self.distance]

      if stupid:
        self.stupidCalculateGradient() # calculate gradient
      else:
        self.makeC()
        self.calculateGradient()
      self.correctParams() # update parameters
    
    # most parameters are cyclic - make them in (0, max)
    for gate in self.gates:
      for basicGate in gate.basicGates:
        for param in basicGate.params:
          param = param.real % (2 * math.pi) 
      gate.time = gate.time.real % (4 * math.pi)
    self.phase = self.phase.real % (2 * math.pi)

    if trackDistance:
      return distances
