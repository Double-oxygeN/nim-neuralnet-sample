import neural1pkg.neuralnet, neural1pkg.activation
import unittest

suite "2-layer Perceptron Test":
  test "type test":
    let
      p0 = newThreeLayerPerceptron(2, 3, 4)
      p1 = newThreeLayerPerceptron(24, 36, 10)
    
    check(p0 is ThreeLayerPerceptron[2, 3, 4])
    check(p1 is ThreeLayerPerceptron[24, 36, 10])
  
  test "feed forward":
    let
      p0 = newThreeLayerPerceptron(3, 2, 6)
      
      o0 = p0.feedForward([1.0, 0.7, 0.6])
    
    echo @o0
  
  test "learning":
    let
      p = newThreeLayerPerceptron(2, 2, 1)
      inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
      targets = [[0.0], [1.0], [1.0], [0.0]]
      goal = 0.001
      ε = 0.03
    
    var
      totalLoss = Inf
      results: array[4, array[1, float]]
      
    while totalLoss > goal:
      totalLoss = 0.0
      
      for i,input in inputs:
        results[i] = p.feedForward(input)
        echo "[",input[0],",",input[1],"] -> ",results[i][0]
        totalLoss += p.getLoss(targets[i])
        p.backProp(targets[i])
      echo "Total Loss: ",totalLoss
    
    for i,input in inputs:
      let r = p.feedForward(input)
      check(targets[i][0] - ε < r[0] and r[0] < targets[i][0] + ε)
        
suite "3-layer Perceptron Test":
  test "type test":
    let
      p0 = newFourLayerPerceptron(3, 5, 4, 2)
      p1 = newFourLayerPerceptron(18, 36, 48, 5)
    
    check(p0 is FourLayerPerceptron[3, 5, 4, 2])
    check(p1 is FourLayerPerceptron[18, 36, 48, 5])
  
  test "feed forward":
    let
      p0 = newFourLayerPerceptron(5, 14, 8, 3)
      
      o0 = p0.feedForward([0.0, 0.2, 0.3, 0.2, 0.9])
    
    echo @o0
  
  test "learning":
    let
      p = newFourLayerPerceptron(2, 2, 2, 1)
      inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
      targets = [[0.0], [1.0], [1.0], [0.0]]
      goal = 0.001
      ε = 0.03
    
    var
      totalLoss = Inf
      results: array[4, array[1, float]]
      
    while totalLoss > goal:
      totalLoss = 0.0
      
      for i,input in inputs:
        results[i] = p.feedForward(input)
        echo "[",input[0],",",input[1],"] -> ",results[i][0]
        totalLoss += p.getLoss(targets[i])
        p.backProp(targets[i])
      echo "Total Loss: ",totalLoss
    
    for i,input in inputs:
      let r = p.feedForward(input)
      check(targets[i][0] - ε < r[0] and r[0] < targets[i][0] + ε)
