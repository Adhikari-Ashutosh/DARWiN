from DARWiN import NN
import numpy as np
def activations():
    works = False
    try:
        NN.argmax(0)
        NN.binstep(0)
        NN.elu(0)
        NN.relu(0)
        NN.linear(0)
        NN.sigmoid(0)
        NN.swish(0)
        NN.lrelu(0)
        NN.tanh(0)
    except:
        works = False
    else:
        works = True
    return works

def randomizedNN():
    works = True
    try:
        testNet = NN.FFN(3,3,5)
        testNet.forward(np.array([1,2,3]))
    except:
        works = False
    return works

def test_activation():
    assert activations() == True

# def test_Net():
#     assert randomizedNN() == True
