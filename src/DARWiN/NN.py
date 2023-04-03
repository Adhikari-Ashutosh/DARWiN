import numpy as np

def binstep(x):
    return 1 if x > 0 else (0 if x == 0 else -1)
    
def linear(x):
    return x
    
def argmax(x):
    return np.argmax(x)
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def tanh(x):
    return np.tanh(x)
    
def relu(x):
    return x if x > 0 else 0
    
def lrelu(x):
    return x if x>0 else 0.1 * x
    
def elu(x):
    return x if x >= 0 else np.exp(x) - 1

def swish(x):
    return x * 1/(1 + np.exp(-x)) 
class FFN:
    def __init__(self , input_size , output_size , layers_sizes , random_state=None):
        
        self.weights = []
        self.biases = []
        self.activations = []
        if random_state:
            np.random.seed(random_state)

        for i in range(1, len(layers_sizes)):
            if i == 1:
                # The first layer weight matrix includes the input size
                self.weights.append(np.random.randn(input_size, layers_sizes[i]))
            elif i == len(layers_sizes) - 1:
                # The last layer weight matrix includes the output size
                self.weights.append(np.random.randn(layers_sizes[i-1], output_size))
            else:
                # Subsequent layers have weights only based on the layer size
                self.weights.append(np.random.randn(layers_sizes[i-1], layers_sizes[i]))
            
            # Biases are added for every layer
            self.biases.append(np.random.randn(layers_sizes[i]))
            self.activations.append(np.random.choice(range(9)))

    def forward(self , x):
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = self.activation(x , i)
        return x

    def activation(self, x, i):
        farr = [binstep , linear , argmax , sigmoid , tanh , relu , lrelu , elu , swish]
        fn = farr[i]
        return fn(x)


        
        



class NEU:

    def __init__(self,Is,Os,Nl,Sl,random_state) -> None:
        '''
        Is Stands for Input size
        Os Stands for Output size
        Nl Stands for Number of Layers
        Sl Stands for Size of Layers
        random_state gives you the power to replicate a NEU
        '''
        self.input = np.array(shape=( 1 , Is+1 ))
        self.layers = FFN(Is,Os,Nl,random_state)
        self.output = np.array(shape = ( 1 , Os ))
    


