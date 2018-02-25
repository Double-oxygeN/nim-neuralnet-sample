## Neuron, known as Perceptron

import random, activation, optimizer

randomize()

type
  Neuron*[N: static[int]] = ref object of RootObj
    weight: array[N, float]
    bias: float
    inputValue: array[N, float]
    outputValue: float
    activator: ActivationFunction
    optimizer: Optimizer

proc newNeuron*(inputNum: static[int], activator: ActivationFunction, optimizer: Optimizer): Neuron[inputNum] =
  ## Create new neuron which has $inputNum inputs.
  ## The result has randomized weights and a bias.
  
  new result
  
  for n in 0..<inputNum:
    result.weight[n] = random(2.0) - 1.0
    result.inputValue[n] = NaN
  result.bias = random(2.0) - 1.0
  result.outputValue = NaN
  result.activator = activator
  result.optimizer = optimizer

proc feedForward*[N: static[int]](self: Neuron[N], input: array[N, float]): float =
  ## Feed forward and get the output.
  
  self.inputValue = input

  result = self.bias
  for i,w in self.weight:
    result += w * input[i]

  result = self.activator.apply(result)
  self.outputValue = result

proc update*[N: static[int]](self: Neuron[N], rate: float): array[N, float] =
  ## Update neuron with gradient descent.
  
  let coefficient = rate * self.activator.grad(self.outputValue)
  
  for i in 0..<N:
    result[i] = self.weight[i] * coefficient
    self.weight[i] = self.optimizer.update(self.weight[i], coefficient * self.inputValue[i])
  self.bias = self.optimizer.update(self.bias, coefficient)

proc getWeight*[N: static[int]](self: Neuron[N]): array[N, float] = self.weight
proc getBias*[N: static[int]](self: Neuron[N]): float = self.bias
proc getOutputValue*[N: static[int]](self: Neuron[N]): float = self.outputValue

proc `$`*[N: static[int]](self: Neuron[N]): string =
  result = "["
  for i in 0..<(N-1):
    result &= $self.weight[i] & " "
  result &= $self.weight[N-1] & "] + " & $self.bias
