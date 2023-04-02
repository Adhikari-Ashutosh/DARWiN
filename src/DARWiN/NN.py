import numpy as np
import random


class FFN:
    def __init__(self, input_size , output_size , layers_sizes, random_state=None):
        self.weights = []
        self.biases = []
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

    def forward(self, x):
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            is_output = True if i == len(self.weights) - 1 else False
            x = self.activation(x, is_output=is_output)
        return x

    def activation(self, x, is_output=False):
        if is_output:
            # Discrete output for output layer
            return np.argmax(x)
        else:
            # Tanh activation for inner layers
            return np.tanh(x)
    def to_hex(self):
        hex_str = ""
        for weights, biases in zip(self.weights, self.biases):
            # convert each weight and bias matrix to a hexadecimal string
            weights_hex = np.array2string(weights.ravel(), separator='')[1:-1].replace(' ', '')
            biases_hex = np.array2string(biases, separator='')[1:-1].replace(' ', '')
            hex_str += weights_hex + biases_hex

        # return the concatenated hexadecimal string
        return hex_str.encode().hex()
    def from_hex(self, hex_str):
        # Split the hex string into weight and bias components
        wb_hex = hex_str.split(" ")

        # Convert each component back to numpy arrays
        weights = []
        biases = []
        for i in range(len(wb_hex)):
            # Determine if this component represents a weight or bias array
            if i % 2 == 0:
                # Weight array
                w_arr = np.frombuffer(bytes.fromhex(wb_hex[i]), dtype=np.float64)
                w_arr = np.reshape(w_arr, (self.layer_sizes[i//2], self.layer_sizes[(i//2)+1]))
                weights.append(w_arr)
            else:
                # Bias array
                b_arr = np.frombuffer(bytes.fromhex(wb_hex[i]), dtype=np.float64)
                biases.append(b_arr)
        
        # Set the model's weights and biases to the decoded values
        self.weights = weights
        self.biases = biases
    def from_hex(self, hex_str):
        # decode the hex string
        hex_bytes = bytes.fromhex(hex_str)
        # split the hex string into weight and bias sub-strings
        sub_hex_strs = [hex_bytes[i:i+8*self.layers_sizes[j]*self.layers_sizes[j+1]+8*self.layers_sizes[j]] for j in range(len(self.layers_sizes)-1)]
        
        # convert each weight and bias sub-string to a numpy array
        for i, sub_hex_str in enumerate(sub_hex_strs):
            weights_hex_str = sub_hex_str[:8*self.layers_sizes[i]*self.layers_sizes[i+1]]
            biases_hex_str = sub_hex_str[8*self.layers_sizes[i]*self.layers_sizes[i+1]:]
            weights_flat = np.frombuffer(bytes.fromhex(weights_hex_str), dtype=np.float64)
            biases_flat = np.frombuffer(bytes.fromhex(biases_hex_str), dtype=np.float64)
            self.weights[i] = np.reshape(weights_flat, (self.layers_sizes[i], self.layers_sizes[i+1]))
            self.biases[i] = np.reshape(biases_flat, (self.layers_sizes[i+1],))
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
    


