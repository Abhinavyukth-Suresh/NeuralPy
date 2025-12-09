import numpy as np

class optimizers():
    def __init__(self,layer):
        self.layer = layer

class VanillaGD():
    def __init__(self,gamma=0.01):
        self.gamma = gamma
        self.name = "Vanilla Gradient Discent"

    def optimize(self,dW,dB,gamma=None):
        if gamma is None:
            gamma = self.gamma
        dW *= gamma
        dB *= gamma
        return dW,dB
    
class SGD():
    def __init__(self,beta = 0.1,gamma=0.01):
        self.gamma = gamma
        self.beta = beta
        self.init = False 
        self.Vdw = None
        self.Vdb = None
        self.name = "Stochastic Gradient Discent"

    def optimize(self,dW,dB,gamma=None,beta=None):
        if gamma is None:
            gamma = self.gamma
        if beta is None:
            beta = self.beta
        if not self.init:
            self.Vdw = np.zeros_like(dW)
            self.Vdb = np.zeros_like(dB)
            self.init = True
        self.Vdw = self.Vdw*beta + dW*(1-beta)
        self.Vdb = self.Vdb*beta + dB*(1-beta)
        dW = self.Vdw*gamma
        dB = self.Vdb*gamma
        return dW,dB

class AdaGrad():
    def __init__(self,epsilon = 0.99,gamma=0.01,inital_learningRate=0.01):
        self.gamma = gamma
        self.init = False
        self.inital_learningRate = inital_learningRate 
        self.epsilon = 0.99
         #adagrab as inital learning rate decreases closer epsilon to 1 
        self.name = "AdaGrad (Adaptive Gradient Descent)"
        

    def optimize(self,dW,dB):

        if not self.init:
            self.eta_w = np.ones_like(dW)*self.inital_learningRate
            self.eta_b = np.ones_like(dB)*self.inital_learningRate
            self.alpha_b = np.zeros_like(dB)
            self.alpha_w = np.zeros_like(dW)
            self.init = True

        dW = dW*self.eta_w
        dB = dB*self.eta_b

        self.alpha_w  = self.alpha_w + dW**2
        self.alpha_b  = self.alpha_b + dB**2

        self.eta_w = self.eta_w/np.sqrt(self.alpha_w + self.epsilon)
        self.eta_b = self.eta_b/np.sqrt(self.alpha_b + self.epsilon)
        return dW,dB
        
