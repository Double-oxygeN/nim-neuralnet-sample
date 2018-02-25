# Copyright 2018 Double_oxygeN
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## SGD Optimizer

import math

type
  Optimizer* = ref object of RootObj
    update: iterator(weight, grad: float): float

proc newOptimizer(update: iterator(weight, grad: float): float): Optimizer =
  new result
  
  result.update = update

proc update*(self: Optimizer, weight, grad: float): float = self.update(weight, grad)

proc SGD*(learningRate: float): Optimizer =
  newOptimizer(iterator(weight, grad: float): float =
    while true:
      yield weight - learningRate * grad)

proc MomentumSGD*(learningRate, momentum: float): Optimizer =
  newOptimizer(iterator(weight, grad: float): float =
    var v = 0.0
    while true:
      v = momentum * v + learningRate * grad
      yield weight - v)

proc AdaGrad*(learningRate: float, epsilon: float = 0.000001): Optimizer =
  newOptimizer(iterator(weight, grad: float): float =
    var G = 0.0
    while true:
      G = G + grad^2
      yield weight - learningRate * grad / sqrt(G + epsilon))

proc AdaDelta*(gamma: float, epsilon: float = 0.000001): Optimizer =
  newOptimizer(iterator(weight, grad: float): float =
    var
      r = 0.0
      s = 0.0
      v = 0.0
    while true:
      r = gamma * r + (1 - gamma) * grad^2
      s = gamma * s + (1 - gamma) * v^2
      v = sqrt(s + epsilon) * grad / sqrt(r + epsilon)
      yield weight - v)
