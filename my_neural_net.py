import numpy as np
import math

"""
 ==================================
 My Neural Network From Scratch
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0,x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x >= 0:
        return 1 
    return 0 
    
def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1
    
class Neural_Network():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
        self.output = 0 #I added this
    def train(self, x1, x2, y):
        inputs = [x1,x2]
        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matrix([[x1*self.input_to_hidden_weights[0,0]+x2*self.input_to_hidden_weights[0,1]],[x1*self.input_to_hidden_weights[1,0]+x2*self.input_to_hidden_weights[1,1]], [x1*self.input_to_hidden_weights[2,0]+x2*self.input_to_hidden_weights[2,1]]])# TODO (3 by 1 matrix)
        hidden_layer_activation = np.zeros((3,1)) 
        for i in range(len(hidden_layer_weighted_input)):# TODO (3 by 1 matrix)

            hidden_layer_activation[i]=rectified_linear_unit(hidden_layer_weighted_input[i]+self.biases[i])
        sum = 0 
        for i in range(3):
            sum+= hidden_layer_activation[i,0]*self.hidden_to_output_weights[0,i]
        
        output = sum 
        activated_output = output # TODO
        
        print("Point:", x1, x2, " Error: ", (0.5)*pow((y - output),2))
        ### Backpropagation ###
        
        # Compute gradients
        output_layer_error = y-output # TODO dC/dy = (y-a)
        #hidden_layer_error = np.matrix([[output_layer_error*rectified_linear_unit_derivative(hidden_layer_weighted_input[i])] for i in range(3)]) # TODO (3 by 1 matrix)


        bias_gradients = np.zeros((3,1))
        for i in range(3):
            
            bias_gradients[i] = self.hidden_to_output_weights[0,i]*rectified_linear_unit_derivative(hidden_layer_activation[i])


        bias_gradients = output_layer_error*bias_gradients
        
       
        hidden_to_output_weight_gradients = output_layer_error*hidden_layer_activation 
        input_to_hidden_weight_gradients = np.zeros((3,2))
        for i in range(3):
            for j in range(2):
                input_to_hidden_weight_gradients[i,j] = self.hidden_to_output_weights[0,i]*rectified_linear_unit_derivative(hidden_layer_activation[i])*inputs[j]
        input_to_hidden_weight_gradients = output_layer_error*input_to_hidden_weight_gradients


        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases + self.learning_rate*bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights + self.learning_rate*input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights + self.learning_rate*hidden_to_output_weight_gradients
        
    def predict(self, x1, x2):
        
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matrix([[x1*self.input_to_hidden_weights[0,0]+x2*self.input_to_hidden_weights[0,1]],[x1*self.input_to_hidden_weights[1,0]+x2*self.input_to_hidden_weights[1,1]], [x1*self.input_to_hidden_weights[2,0]+x2*self.input_to_hidden_weights[2,1]]])# TODO (3 by 1 matrix)
 
        hidden_layer_activation = np.zeros((3,1)) 

        for i in range(len(hidden_layer_weighted_input)):# TODO (3 by 1 matrix)
            hidden_layer_activation[i]=rectified_linear_unit(hidden_layer_weighted_input[i]+self.biases[i])
        summ = 0 
        for i in range(3):
            summ+= hidden_layer_activation[i,0]*self.hidden_to_output_weights[0,i]
        output = summ
        activated_output = output 
        
        return activated_output.item()
    
    
    # Run this to train my neural network once complete the train method
    def train_neural_network(self):
        
        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:                
                self.train(x[0], x[1], y)
    
    # Run this to test my neural network implementation for correctness after it is trained
    def test_neural_network(self):
        
        for point in self.testing_points:

            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return
        
x = Neural_Network()
x.train_neural_network()
# UNCOMMENT THE LINE BELOW TO TEST MY NEURAL NETWORK
x.test_neural_network()  
