from neural_network import NeuralNetwork
from lyapunov_learning import LyapunovLearning
from system_dynamics import  SystemDynamics
import torch.nn.functional as F
import torch
import numpy as np
import timeit 
import matplotlib.pyplot as plt
from dreal import *
import sympy as sy

#------------------------------------------------------------------------
# This is an example for a Lyapunov function learning process 
# for inverted pendulum dynamics with input constraints 
#------------------------------------------------------------------------

N = 400                                         # sample size
torch.manual_seed(10)  
input_data = torch.Tensor(N, 2).uniform_(-6, 6) # uniform distribution         
x_0 = torch.zeros([1, 2])                       # initial state values  
x1 = Variable("x1")
x2 = Variable("x2")
system_states = [x1,x2]
dynamic_model = SystemDynamics(G=9.81, L=0.5, m=0.15, b=0.1)
learning_lyapunov = LyapunovLearning()
valid = False

while not valid: 
    model = NeuralNetwork(n_input=2,n_hidden=6,n_output=1,
                          initial_control=torch.tensor([[-23.58639732,  -5.31421063]]),
                          control_max=100, control_min=-50)
    loss_values = []
    epoch = 0 
    total_time_sum = 0
    epochs = 2000
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = timeit.default_timer()
    while epoch < epochs and not valid: 
      # network output
      V_candidate, u = model(input_data) 
      V0,u0 = model(x_0)
      # system model 
      f = dynamic_model.system(input_data,u)
      # ∑∂V/∂xᵢ*fᵢ
      L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(model.dtanh(V_candidate),model.second_layer.weight)\
                          *model.dtanh(torch.tanh(torch.mm(input_data,model.first_layer.weight.t())+model.first_layer.bias)),model.first_layer.weight),f.t()),0)
      #------------------------------------------------------------------
      # loss function. note: relu function is max{0,x} function 
      # control input constraints umin <= u <= umax
      #  umin <= u: F.relu(u-umin)
      #  u<= umax : F.relu(u+umaxn)
      emperical_risk = (F.relu(-V_candidate)+ 1.5*F.relu(L_V+0.5)).mean()\
                  +2.2*((learning_lyapunov.norm_l2(input_data)-6*V_candidate).pow(2)).mean()+(V0).pow(2) + (F.relu(-u+100) + F.relu(u+50)).mean()
      #------------------------------------------------------------------            
      print('Epoch %2s/%s \n [==================] Empirical risk: %10.6f' % (epoch, epochs,emperical_risk.item())) 
      loss_values.append(emperical_risk.item())
      optimizer.zero_grad()
      emperical_risk.backward()
      optimizer.step() 
      control = model.control.weight.data.numpy()
      # verification of Lyapunov conditions each 10 epochs 
      if epoch % 10 == 0:                 
          f = dynamic_model.system_expression(x1, x2, control=control)
          V_learn = learning_lyapunov.lyapunov_candidate_construction(model=model,
                                                                      system_states=system_states,
                                                                      activation_function="tanh")         
          input_data, total_time, valid = learning_lyapunov.verification(system_states=system_states,
                                                             system_dynamic=f, 
                                                             lyapunov_candidate=V_learn,
                                                             lower_boundary=0.5,
                                                             upper_boundary=6,
                                                             precision=1e-5,
                                                             epsilon=1e-5,
                                                             input_data=input_data)
          total_time_sum += total_time
      epoch += 1
      stop = timeit.default_timer()
    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", total_time_sum)
    print("Control: ",model.control.weight.data)


# plot function 
numpoints = 100                                   # define resolution
domain=6                                          # {xᵢ|-6≤ xᵢ ≤6}
# define plotting range and mesh
x = np.linspace(-domain, domain, numpoints)
y = np.linspace(-domain, domain, numpoints)
X, Y = np.meshgrid(x, y)
V = sy.lambdify([x1,x2], V_learn, "numpy")
learning_lyapunov.Plot_function(X, Y, V(X,Y),xlabel="x",ylabel="y",zlabel="V(x)")

# 2D
#plt.plot(V(x,y))
#plt.grid(True)