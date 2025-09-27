import numpy as np

class Activations():

    def __init__(self,type=None):
        self.type = type
        self.activation = None
        self.l = 0.1
        self.pairs = {'sigmoid'  :(self.sigmoid,self.d_sigmoid),
                      'tanh'     :(self.tanh,   self.d_tanh   ),
                      'relu'     :(self.relu,   self.d_relu   ),
                      'softmax'  :(self.softmax,self.d_softmax),
                      'leakyRelu':(self.LeakyRelu,self.d_LeakyRelu)
                      }
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def d_sigmoid(self,x,e):
        return x*(1-x)*e

    def tanh(self,x):
        return np.tanh(x)

    def d_tanh(self,x,e):
        return e-x*x*e
    # (1-x**2)*e

    def relu(self,x):
        self.X = x.copy()
        x[x<=0]=0
        return x
    
    def d_relu(self,x,e):
        x[self.X<=0]=0
        x[self.X>0]=1
        return x*e

    def LeakyRelu(self,x):
        self.X = x
        x[x<=0]*= self.l
        return x
    
    def d_LeakyRelu(self,x,e):
        x[self.X<=0]=self.l
        x[self.X>0]=1
        return x*e
    
    def softmax(self,x):
        ex = np.exp(x-np.max(x))
        return ex/np.sum(ex,axis=1).reshape(-1,1)
    
    def d_softmax(self,x,e):
        k,j = x.shape
        X = x.reshape(k,1,j)
        i = np.identity(x.shape[1])
        idty = np.array([x[t]*i - (X[t].T@X[t]) for t in range(k)])
        de = np.einsum('ijk,ij->ik',idty,e)
        return de
   
