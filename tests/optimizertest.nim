import neural1pkg.optimizer
import unittest

suite "Optimizer Test":
  test "type test":
    
    check(SGD(0.2) is Optimizer)
    check(MomentumSGD(learningRate = 0.2, momentum = 0.9) is Optimizer)
    check(AdaGrad(learningRate = 0.2) is Optimizer)
    check(AdaDelta(gamma = 0.9) is Optimizer)
  
  test "iteration test":
    let
      o0 = AdaGrad(learningRate = 0.2)
      o1 = AdaDelta(gamma = 0.9)
    
    echo "AdaGrad"
    for i in 1..100:
      echo o0.update(1.0, 0.3)
    
    echo "AdaDelta"
    for i in 1..100:
      echo o1.update(1.0, 0.3)
