## Neural Net

import
  layer,
  activation,
  optimizer

type
  ThreeLayerPerceptron*[I, H, O: static[int]] = ref object of RootObj
    l0: Layer[I, H]
    l1: Layer[H, O]
  
  FourLayerPerceptron*[I, H0, H1, O: static[int]] = ref object of RootObj
    l0: Layer[I, H0]
    l1: Layer[H0, H1]
    l2: Layer[H1, O]

proc newThreeLayerPerceptron*(input, hidden, output: static[int], activator: ActivationFunction = sigmoid, optimizer: Optimizer = SGD(0.2)): ThreeLayerPerceptron[input, hidden, output] =
  new result

  result.l0 = newLayer(input, hidden, activator, optimizer)
  result.l1 = newLayer(hidden, output, activator, optimizer)

proc feedForward*[I, H, O: static[int]](self: ThreeLayerPerceptron[I, H, O], input: array[I, float]): array[O, float] =
  let
    h0 = self.l0.feedForward(input)
  
  return self.l1.feedForward(h0)

proc getLoss*[I, H, O: static[int]](self: ThreeLayerPerceptron[I, H, O], target: array[O, float]): float =
  return self.l1.calcRSS(target)

proc backProp*[I, H, O: static[int]](self: ThreeLayerPerceptron[I, H, O], target: array[O, float]) =
  let
    r1 = self.l1.updateOutputLayer(target)
  discard self.l0.updateHiddenLayer(r1)

proc newFourLayerPerceptron*(input, hidden0, hidden1, output: static[int], activator: ActivationFunction = sigmoid, optimizer: Optimizer = SGD(0.2)): FourLayerPerceptron[input, hidden0, hidden1, output] =
  new result
  
  result.l0 = newLayer(input, hidden0, activator, optimizer)
  result.l1 = newLayer(hidden0, hidden1, activator, optimizer)
  result.l2 = newLayer(hidden1, output, activator, optimizer)

proc feedForward*[I, H0, H1, O: static[int]](self: FourLayerPerceptron[I, H0, H1, O], input: array[I, float]): array[O, float] =
  let
    h0 = self.l0.feedForward(input)
    h1 = self.l1.feedForward(h0)

  return self.l2.feedForward(h1)

proc getLoss*[I, H0, H1, O: static[int]](self: FourLayerPerceptron[I, H0, H1, O], target: array[O, float]): float =
  return self.l2.calcRSS(target)

proc backProp*[I, H0, H1, O: static[int]](self: FourLayerPerceptron[I, H0, H1, O], target: array[O, float]) =
  let
    r2 = self.l2.updateOutputLayer(target)
    r1 = self.l1.updateHiddenLayer(r2)
  discard self.l0.updateHiddenLayer(r1)
