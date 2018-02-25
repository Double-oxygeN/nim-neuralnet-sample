## Activation Functions

import math

type
  ActivationFunction* = ref object of RootObj
    apply: (proc (x: float): float)
    grad: (proc (fx: float): float)

proc newActivationFunction(apply: proc (x: float): float, grad: proc (fx: float): float): ActivationFunction =
  new result
  
  result.apply = apply
  result.grad = grad

proc apply*(self: ActivationFunction, x: float): float =
  self.apply(x)

proc apply*[N: static[int]](self: ActivationFunction, xs: array[N, float]): array[N, float] =
  for i,x in xs:
    result[i] = self.apply(x)

proc grad*(self: ActivationFunction, fx: float): float =
  self.grad(fx)

let
  identity* = newActivationFunction(
    apply = proc (x: float): float = x,
    grad = proc (fx: float): float = 1)
  
  sigmoid* = newActivationFunction(
    apply = proc (x: float): float = 1 / (1 + exp(-x)),
    grad = proc (fx: float): float = fx * (1 - fx))
    
  tanh* = newActivationFunction(
    apply = proc (x: float): float = math.tanh(x),
    grad = proc (fx: float): float = 1 - fx ^ 2)
  
  relu* = newActivationFunction(
    apply = proc (x: float): float = max(0, x),
    grad = proc (fx: float): float = (if fx == 0: 0 else: 1))

  leakyrelu* = newActivationFunction(
    apply = proc (x: float): float = (if x < 0: 0.01 * x else: x),
    grad = proc (fx: float): float = (if fx < 0: 0.01 else: 1))
