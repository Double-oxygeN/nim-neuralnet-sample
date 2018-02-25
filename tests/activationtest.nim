import neural1pkg.activation
import unittest, fenv

suite "Activation Function Test":
  test "type test":
    
    check(identity is ActivationFunction)
    check(sigmoid is ActivationFunction)
    check(tanh is ActivationFunction)
    check(relu is ActivationFunction)
    check(leakyrelu is ActivationFunction)
  
  test "apply test":
    let
      ε = 2 * epsilon(float)
      
      functions = [identity, sigmoid, tanh, relu, leakyrelu]
      xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
      ys = [
        [-2.0, 0.1192029220221175, -0.9640275800758169, 0.0, -0.02],
        [-1.0, 0.2689414213699951, -0.7615941559557649, 0.0, -0.01],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [1.0, 0.7310585786300049, 0.7615941559557649, 1.0, 1.0],
        [2.0, 0.8807970779778823, 0.9640275800758169, 2.0, 2.0]
      ]
    
    echo "ε = ", ε
    
    for i,x in xs:
      for j,f in functions:
        let
          y = ys[i][j]
          fx = f.apply(x)
        check(y - ε < fx and fx < y + ε)
    
