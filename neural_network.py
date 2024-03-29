import torch
from dreal import *

class NeuralNetwork(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output,initial_control, control_max, control_min):
        """
        Args: 
            n_input (int): number of inputs
            n_hidden (int): number of neurons in hidden layer
            n_output (int): number of neuron in output layer
            initial_control (list): initial control law
            control_max (int): maximal control value 
            control_min (int): minimal control value

        """
        super(NeuralNetwork, self).__init__()
        torch.manual_seed(2)
        # create first layer with linear activation function 
        self.first_layer = torch.nn.Linear(n_input, n_hidden) 
        # create second layer with linear activation function
        self.second_layer = torch.nn.Linear(n_hidden,n_output)
        # create control layer with linear activation function
        self.control = torch.nn.Linear(n_input,1,bias=False)
        # set control layer weight with initial control law 
        self.control.weight = torch.nn.Parameter(initial_control)
        self.control_max = control_max
        self.control_min = control_min

    def forward(self, x):
        """
        Args: 
            x: system dataset 
        Return: 
            lyapunov_candidate: Lyapunov candidate function V(x) for system x
            control_function: Possible control function u(x) for system x
        """
        activation_function = torch.nn.Tanh()

        # Apply input constraints to the input data
        constrained_x = torch.clamp(x, self.control_min, self.control_max)

        x_i = activation_function(self.first_layer(constrained_x))
        lyapunov_candidate = activation_function(self.second_layer(x_i))

        # Apply input constraints to the control function
        constrained_control = torch.clamp(self.control(constrained_x), self.control_min, self.control_max)
        control_function = activation_function(constrained_control)

        return lyapunov_candidate, control_function

  
    def derivative(self,x):
        """
        Args: 
            x: variable 
        Return: 
            derivative of tanh(x)
        """
        # Derivative of activation
        return 1.0 - x**2
        
    def umax(self, signal):
        """
        Args: 
            singal (tensor): control signal to be limited  

        Return:
            singal (tensor): control limeted 
        """

        for index in range(0,len(signal)):
            if signal[index]>self.control_max:
                signal[index]=self.control_max

        return signal
