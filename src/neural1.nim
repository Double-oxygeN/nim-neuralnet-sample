import
  neural1pkg.neuralnet,
  neural1pkg.optimizer
import
  random

randomize()

let
  net = newThreeLayerPerceptron(2, 24, 2, optimizer = AdaDelta(gamma = 0.9))
  
  inputData = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
  trainData = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


when isMainModule:
  for count in 1..500_000:
    let
      n = random(inputData.len())
      result = net.feedForward(inputData[n])
    
    if count mod 10_000 == 0:
      echo "[", count, "] input: (", inputData[n][0], ",", inputData[n][1], "), output: (", result[0], ",", result[1], "), loss: ", net.getLoss(trainData[n])
    
    net.backProp(trainData[n])
  
  var
    totalLoss = 0.0
  
  echo "Final report:"
  for n,input in inputData:
    let result = net.feedForward(input)
    echo "input: (", inputData[n][0], ",", inputData[n][1], "), output: (", result[0], ",", result[1], ")"
    totalLoss += net.getLoss(trainData[n])
  echo "Total Loss: ", totalLoss
