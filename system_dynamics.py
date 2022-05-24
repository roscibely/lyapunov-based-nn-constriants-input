
import numpy as np
import torch
from dreal import *

class SystemDynamics():

    def __init__(self, G, L, m, b):
        """
        Args: 
            G (float): gravity
            L (float):length of the pole 
            m (float): ball mass
        """
        self.G = G     
        self.L =L
        self.m = m
        self.b = b

    def system(self, x,u):
        """
        Args: 
            x (tensor): system states 
            u (list): control  
        Retrun: 
            y : dynamics
        """
        y = []
        for r in range(0,len(x)): 
            f = [ x[r][1], 
                (self.m*self.G*self.L*np.sin(x[r][0])- self.b*x[r][1]) / (self.m*self.L**2)]
            y.append(f) 
        y = torch.tensor(y)
        y[:,1] = y[:,1] + (u[:,0]/(self.m*self.L**2))
        return y

    def system_expression(self, x1, x2, control):
        """
        Agrs: 
            x1: pendulum position 
            x2: pendulum angle
            control: control signal 
        Return: 
            f: system output  
        """

        # network control
        u_NN = (control.item(0)*x1 + control.item(1)*x2) 
        # dynamic
        f = [ x2,(self.m*self.G*self.L*sin(x1) + u_NN - self.b*x2) /(self.m*self.L**2)]

        return f