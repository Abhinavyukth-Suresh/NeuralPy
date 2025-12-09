import numpy as np
from layer import*
import optimizers

class Dense(Layer):
                    #####  DENSE LAYER   #####
    def __init__(self,n_nodes,input_shape,activation='tanh',use_bias=True,
                 weights_initializer = 'auto',bias_initializer = 'auto',
                 optimizer = optimizers.VanillaGD(gamma = 0.01)):

        #DENSE PARAMETERS
        self.type = 'Dense'
        n_features =  input_shape[1]
        self.n_params = n_nodes*n_features + n_nodes
        self.activation = activation.lower()
        self.active,self.d_active = self.assign_activation(activation)
        self.input_shape = input_shape
        self.output_shape = (None,n_nodes)
        self.optim = optimizer.name

        self.__inputs = None
        self.__outputs = None

        self.optimizer = optimizer
        self.weights,self.bias = self.Weights_and_Bias_Initializer(weights_initializer,bias_initializer,n_nodes,n_features,activation)
        self.dW = np.zeros_like(self.weights)
        self.dB = np.zeros_like(self.bias)
        self.pred = np.frompyfunc(self.predict,1,1)

    #FEED FORWARD
    def predict(self,inputs):
        self.__inputs = inputs
        self.__outputs = self.active(np.dot(inputs,self.weights)+self.bias)
        return self.__outputs

    #GRADIENTS
    def gradients(self,error):
        dO = self.d_active(self.__outputs,error)
        dE = np.dot(dO,self.weights.T)
        dW = np.dot(self.__inputs.T,dO)
        dB = np.expand_dims(np.sum(dO,axis=0),axis=0)
        dW,dB = self.optimizer.optimize(dW,dB)
        self.dW += dW
        self.dB += dB
        return dE

    # INPUT GRADIENTS WRT ERROR
    def input_Grad(self,error):
        dO = self.d_active(self.output,error)
        return np.dot(dO,self.weights.T)
 
    # APPLYING THE GRADIENTS
    def Apply_Grad(self):
        self.weights += self.dW
        self.bias += self.dB
        self.dW = np.zeros_like(self.weights)
        self.dB = np.zeros_like(self.bias)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    from optimizers import *
    inputs = np.random.random((10,3))
    outputs = inputs
    
    l = Dense(3,(None,3),activation='tanh')
    l.optimizer = AdaGrad()

    h = []
    t=time.time()
    N = 300
    for i in range(N):
        p = l.predict(inputs)
        
        e = outputs-p
        l.gradients(e)
        l.Apply_Grad()
        h.append(np.mean(e))
    print(time.time()-t)
    plt.plot(h)
    

    l = Dense(3,(None,3),activation='tanh')
    l.optimizer = SGD()

    h = []
    t=time.time()
    for i in range(N):
        p = l.predict(inputs)
      
        e = outputs-p
        
        l.gradients(e)
        l.Apply_Grad()
        h.append(np.mean(e))
    print(time.time()-t)


    plt.plot(h,color='black')
    plt.show()     
