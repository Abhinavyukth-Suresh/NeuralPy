import numpy as np
from NeuralPy.Activation_functions import*

######  ERRORS   ######

class Errors():
    def __init__(self):
        self.Errors = {"MSE":(self.dmse,self.mse),"Entropy":(self.d_entropy,self.entropy)}
        
    def mse(self,pred,output):
        return (output-pred)**2
    
    def dmse(self,pred,output):
        return (output-pred)

    def entropy(self,pred,output):
        raise NotImplementedError()
    
    def d_entropy(self,pred,output):
        raise NotImplementedError()
