import numpy as np
from Activations import *


class Layer():
    def __init__(self):
        self.type = None
        self.n_parans = 0
        self.inputs_shape = None
        self.output_shape = None
        self.active, self.d_active = None, None
        self.solver = None

    def Weights_and_Bias_Initializer(self,weights_initializer,bias_initializer,n_nodes,n_features,activation=None):
        # INITIALIZING WEIGHTS
        W_init = weights_initializer
        if W_init == 'auto':
            if activation.lower() == 'sigmoid':
                W_init = 'Xavier_uniform'
                
            elif activation.lower() == 'relu' or activation.lower() == 'leakyrelu':
                W_init = 'He_uniform'

            elif activation.lower() == 'tanh':
                W_init = 'Xavier_uniform'
        
        if W_init == 'uniform': #uniform distribution
            a = 1/np.sqrt(n_features)
            weights = np.random.uniform(-a,a,(n_features,n_nodes))
            
        elif W_init == 'Xavier_normal' or 'Gorat_normal':
            a = np.sqrt(2/(n_features+n_nodes))
            weights = np.random.normal(0,a,(n_features,n_nodes))
            
        elif W_init == 'Xavier_uniform' or 'Gorat_uniform':
            a = np.sqrt(6/(n_features+n_nodes))
            self.weights = np.random.uniform(-a,a,(n_features,n_nodes))

        elif W_init == 'He_uniform':
            a = np.sqrt(6/(n_features))
            weights = np.random.uniform(-a,a,(n_features,n_nodes))
            
        elif W_init == 'He_noraml':
            a = np.sqrt(2/(n_features))
            weights = np.random.normal(0,a,(n_features,n_nodes))
        else:
            raise NameError(f" weight initializer {W_init} is not implemented")

        #INITIALIZING BIAS
        B_init = bias_initializer
        if B_init == 'auto':
            if activation.lower() == 'sigmoid':
                B_init = 'Xavier_uniform'
                
            elif activation.lower() == 'relu':
                B_init = 'He_uniform'

            elif activation.lower() == 'tanh':
                B_init = 'Xavier_uniform'

        if B_init == 'uniform': #uniform distribution
            a = 1/np.sqrt(n_features)
            bias = np.random.uniform(-a,a,(1,n_nodes))
            
        elif B_init == 'Xavier_normal' or 'Gorat_normal':
            a = np.sqrt(2/(n_features+n_nodes))
            bias = np.random.normal(0,a,(1,n_nodes))
            
        elif B_init == 'Xavier_uniform' or 'Gorat_uniform':
            a = np.sqrt(6/(n_features+n_nodes))
            bias = np.random.uniform(-a,a,(1,n_nodes))

        elif B_init == 'He_uniform':
            a = np.sqrt(6/(n_features))
            bias = np.random.uniform(-a,a,(1,n_nodes))
            
        elif B_init == 'He_noraml':
            a = np.sqrt(2/(n_features))
            bias = np.random.normal(0,a,(1,n_nodes))
        else:
            raise NameError(f" weight initializer {W_init} is not implemented")

        return weights,bias

    #INITIALIZING KERNAL FOR CONVOLUTION
    def kernal_bias_initializer(self,kernal_initializer,bias_initializer,fshape,nfilters,activation='none'):
        #self,weights_initializer,bias_initializer,n_nodes,n_features,activation=None
        
        # INITIALIZING WEIGHTS
        (fx,fy,fs,nf) = fshape
        W_init = kernal_initializer
        if W_init == 'auto':
            if activation.lower() == 'sigmoid':
                W_init = 'Xavier_uniform'
                
            elif activation.lower() == 'relu' or activation.lower() == 'leakyrelu':
                W_init = 'He_uniform'

            elif activation.lower() == 'tanh':
                W_init = 'Xavier_uniform'
        
        if W_init == 'uniform': #uniform distribution
            a = 1/np.sqrt(nf)
            weights = np.random.uniform(-a,a,fshape)
            
        elif W_init == 'Xavier_normal' or 'Gorat_normal':
            a = np.sqrt(2/(nf+fx+fy+fs))
            weights = np.random.normal(0,a,fshape)
            
        elif W_init == 'Xavier_uniform' or 'Gorat_uniform':
            a = np.sqrt(6/(nf+fx+fy+fs))
            self.weights = np.random.uniform(-a,a,fshape)

        elif W_init == 'He_uniform':
            a = np.sqrt(6/(fs))
            weights = np.random.uniform(-a,a,fshape)
            
        elif W_init == 'He_noraml':
            a = np.sqrt(2/(nf))
            weights = np.random.normal(0,a,fshape)
        else:
            raise NameError(f" weight initializer {W_init} is not implemented")

        #INITIALIZING BIAS
        B_init = bias_initializer
        if B_init == 'auto':
            if activation.lower() == 'sigmoid':
                B_init = 'Xavier_uniform'
                
            elif activation.lower() == 'relu':
                B_init = 'He_uniform'

            elif activation.lower() == 'tanh':
                B_init = 'Xavier_uniform'

        if B_init == 'uniform': #uniform distribution
            a = 1/np.sqrt(nf)
            bias = np.random.uniform(-a,a,(1,nfilters))
            
        elif B_init == 'Xavier_normal' or 'Gorat_normal':
            a = np.sqrt(2/(nf+fs+fx+fy))
            bias = np.random.normal(0,a,(1,nfilters))
            
        elif B_init == 'Xavier_uniform' or 'Gorat_uniform':
            a = np.sqrt(6/(nf+fx+fy+fs))
            bias = np.random.uniform(-a,a,(1,nfilters))

        elif B_init == 'He_uniform':
            a = np.sqrt(6/(nf+fx+fy))
            bias = np.random.uniform(-a,a,(1,nfilters))
            
        elif B_init == 'He_noraml':
            a = np.sqrt(2/(nf+fx+fy))
            bias = np.random.normal(0,a,(1,nfilters))
        else:
            raise NameError(f" weight initializer {W_init} is not implemented")

        return weights,bias #raise NotImplementedError()

    #ASSIGNING ACTIVATION FUNCTIONS
    def assign_activation(self,a):
        Act = Activations()
        if a not in list(Act.pairs):
            raise NameError(f"activation function ({a}) is not defined")
        return Act.pairs[a]

    #FEEDFORWARD
    def predict(self,inputs):
        raise NotImplementedError()

    #GRADIENTS
    def gradients(self,error):
        raise NotImplementedError()

    #APPLY THE GRADIENTS
    def Apply_Grad(self,dW,dB):
        raise NotImplementedError()
    
