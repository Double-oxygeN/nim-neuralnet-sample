import neural1pkg.layer, neural1pkg.activation, neural1pkg.optimizer
import unittest, typetraits

suite "Layer Test":
  test "type test":
    let
      l0 = newLayer(13, 5, sigmoid, SGD(0.2))
      l1 = newLayer(2, 2, sigmoid, SGD(0.2))
      # l2 = newLayer(2000, 4000)
    
    check(l0.type.name == "Layer[13, 5]")
    check(l1.type.name == "Layer[2, 2]")
    # check(l2.type.name == "Layer[2000, 4000]")
  
  test "feed forward":
    let
      l0 = newLayer(4, 5, sigmoid, SGD(0.2))
      l1 = newLayer(5, 2, sigmoid, SGD(0.2))
      
      h1 = l0.feedForward([1.0, 1.0, 0.0, 0.2])
      h2 = l1.feedForward(h1)
    
    echo @h1
    echo @h2
