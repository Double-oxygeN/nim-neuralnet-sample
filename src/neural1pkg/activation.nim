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
