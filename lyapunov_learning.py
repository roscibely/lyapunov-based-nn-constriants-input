
from dreal import *
from dreal import tanh
import torch 
import torch.nn.functional as F
import timeit 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class LyapunovLearning():

    def __init__(self):
        pass

    
    def smt_check(self, system_states, system_dynamic, lyapunov_candidate, lower_boundary, upper_boundary, precision=1e-2, epsilon=1e-5):  
        """
        Args: 
           system_states: system states 
           system_dynamic: dynamical system 
           lyapunov_candidate: candidate Lyapunov function V
           lower_boundary: 
           upper_boundary: 
        Return: 
            False: if there is no state violating the conditions
        """
        #Configuration for verifying with dreal
        config = Config()
        config.use_polytope_in_forall = True
        config.use_local_optimization = True
        config.precision = precision

        domain= Expression(0)
        derivative_of_lyapunov_candidate = Expression(0)
        for i in range(len(system_states)):
            domain += system_states[i]*system_states[i]
            derivative_of_lyapunov_candidate += system_dynamic[i]*lyapunov_candidate.Differentiate(system_states[i])  
        domain_in_bound = logical_and(lower_boundary*lower_boundary <= domain, domain <= upper_boundary*upper_boundary)
     
        condition = logical_and(logical_imply(domain_in_bound, lyapunov_candidate >= 0),
                            logical_imply(domain_in_bound, derivative_of_lyapunov_candidate<= epsilon))
        return CheckSatisfiability(logical_not(condition),config)

    def counterexample(self, x,counterex,sample): 
        """
        Args: 
            x: system states 
            counterex: counterexample
            sample: number of samples 
        Return: 
            x: system states
        """
        c = []
        nearby= []
        for i in range(counterex.size()):
            c.append(counterex[i].mid())
            lb = counterex[i].lb()
            ub = counterex[i].ub()
            nearby_ = np.random.uniform(lb,ub,sample)
            nearby.append(nearby_)
        for i in range(sample):
            n_pt = []
            for j in range(x.shape[1]):
                n_pt.append(nearby[j][i])       
            x = torch.cat((x.float(), torch.tensor([n_pt]).float()), 0).float()
        return x


    def norm_l2(self, system_states):
        """
        Args: 
            system_states: system states 
        Return: 
            y: returns L2 norm (Euclidean distance)  ||x||²
        """
        # Circle function values
        y = []
        for r in range(0,len(system_states)):
            v = 0 
            for j in range(system_states.shape[1]):
                v += system_states[r][j]**2
            f = [torch.sqrt(v)]
            y.append(f)
        y = torch.tensor(y)
        return y
    
    def Plot_function(self,X, Y, V,xlabel,ylabel,zlabel):
        """
        Args:
            X: grid
            Y: grid
            V: Lyapunov function
            xlabel: x-axis label
            ylabel: y-axis label
            zlabel: z-axis label
        Returns:
            ax: object
        """
        plt.rcParams.update({'font.size': 14})  
        fig = plt.figure()
        fig.set_size_inches(9,6)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,V, rstride=1, cstride=1, alpha=0.5, cmap=cm.RdGy)
        ax.contour(X,Y,V,10, zdir='z', offset=0, cmap=cm.RdGy)            # contour
        ax.set_xlabel(xlabel,fontsize=20)
        ax.set_ylabel(ylabel,fontsize=20)
        ax.set_zlabel(zlabel,fontsize=20)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        return ax

    def verification(self, system_states, system_dynamic, lyapunov_candidate,
     lower_boundary, upper_boundary, precision, epsilon, input_data):
        """
        Args: 
           system_states: system states 
           system_dynamic: dynamical system 
           lyapunov_candidate: candidate Lyapunov function V
           lower_boundary: 
           upper_boundary: 
           precision: 
           epsilon: 
        Return: 
            total_time:
            valid: 
        """
        start_ = timeit.default_timer() 
        result= self.smt_check(system_states=system_states, system_dynamic=system_dynamic, lyapunov_candidate=lyapunov_candidate,
            lower_boundary=lower_boundary, upper_boundary=upper_boundary, precision=precision, epsilon=epsilon)
        stop_ = timeit.default_timer() 
        if (result): 
            valid = False
            input_data = self.counterexample(x=input_data,counterex=result,sample=10)
        else: 
            print("Found a Lyapunov function.") 
            valid = True
            print(lyapunov_candidate, " is a Lyapunov function.")
        total_time = (stop_ - start_)

        return input_data, total_time, valid 

    def lyapunov_candidate_construction(self, model, system_states, activation_function="tanh"):

        """
        Args: 
            model: Neural network model
            system_states: system states x1, x2, ...xn
            activation_function: activation function for construction of Lyapunov function
        Return: 
            V_learn: Lyapunov function candidate
        """

        # model weights 
        w1 = model.first_layer.weight.data.numpy()
        w2 = model.second_layer.weight.data.numpy()
        b1 = model.first_layer.bias.data.numpy()
        b2 = model.second_layer.bias.data.numpy()

        # Candidate V(x) = tanh(∑w.x +b2)
        if activation_function=="tanh":
            z1 = np.dot(system_states,w1.T)+b1
            a1 = []
            for j in range(0,len(z1)):
                a1.append(tanh(z1[j]))
            z2 = np.dot(a1,w2.T)+b2
            V_learn = tanh(z2.item(0))

            return V_learn
        else: 
            raise ValueError('unknown function')