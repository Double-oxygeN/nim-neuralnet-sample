import neural1pkg.neuron, neural1pkg.activation, neural1pkg.optimizer
import unittest, typetraits

suite "Neuron Test":
  test "neuron type":
    let
      n0 = newNeuron(8, sigmoid, SGD(0.2))
      n1 = newNeuron(2, sigmoid, SGD(0.2))
      n2 = newNeuron(3000, sigmoid, SGD(0.2))
    
    check(n0.type.name == "Neuron[8]")
    check(n1.type.name == "Neuron[2]")
    check(n2.type.name == "Neuron[3000]")
    
  test "feed forward":
    let
      n0 = newNeuron(3, sigmoid, SGD(0.2))
      n1 = newNeuron(5, sigmoid, SGD(0.2))
    
    echo n0
    echo n0.feedForward([1.0, 0.2, 0.7])
    
    echo n1
    echo n1.feedForward([0.0, 0.1, 0.9, 1.0, 0.5])
    
    check(n0.feedForward([1.0, 0.2, 0.7]) is float)
    check(n1.feedForward([0.0, 0.1, 0.9, 1.0, 0.5]) is float)
