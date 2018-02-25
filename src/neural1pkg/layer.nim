## Layer

import neuron, activation, optimizer
import math

type
  Layer*[I, O: static[int]] = array[O, Neuron[I]]

proc newLayer*(input, output: static[int], activator: ActivationFunction, optimizer: Optimizer): Layer[input, output] =
  for i in 0..<output:
    result[i] = newNeuron(input, activator, optimizer)

proc feedForward*[I, O: static[int]](self: Layer[I, O], input: array[I, float]): array[O, float] =
  for i in 0..<O:
    result[i] = self[i].feedForward(input)

proc updateOutputLayer*[I, O: static[int]](self: Layer[I, O], target: array[O, float]): array[I, float] =
  ## Update output layer.
  
  for i in 0..<I:
    result[i] = 0.0

  for i,t in target:
    for j,q in self[i].update(self[i].getOutputValue() - t):
      result[j] += q

proc updateHiddenLayer*[I, O: static[int]](self: Layer[I, O], rate: array[O, float]): array[I, float] =
  ## Update hidden layer.
  
  for i in 0..<I:
    result[i] = 0.0
  
  for i,r in rate:
    for j,q in self[i].update(r):
      result[j] += q

proc calcRSS*[I, O: static[int]](self: Layer[I, O], target: array[O, float]): float =
  ## Calculate RSS (Residual Sum of Squares).
  
  result = 0.0
  for i,t in target:
    result += (self[i].getOutputValue() - t) ^ 2
