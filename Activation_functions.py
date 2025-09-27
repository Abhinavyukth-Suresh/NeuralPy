import numpy as np

##########   ACTIVATION FUNCTIONS   ############
'''
def sigmoid():
    return lambda x: 1/(1+np.exp(-x))

def dsigmoid():
    return lambda x: x*(1-x)

def tanh():
    return lambda x: np.tanh(x)

def dtanh():
    return lambda x: 1-(x**2)
'''
def softmax(x):
    ex = np.exp(x-np.max(x))
    return ex/np.sum(ex,axis=1).reshape(-1,1)

def dsoftmax(x,p):
    k,j = x.shape
    X = x.reshape(k,1,j)
    i = np.identity(x.shape[1])
    idty = np.array([x[t]*i - (X[t].T@X[t]) for t in range(k)])
    #de =  np.tensordot(idty ,p ,axes = ((1),(1)))
    de = np.einsum('ijk,ij->ik',idty,p)
    return de
    
  
def sigmoid(x):
    return  1/(1+np.exp(-x))

def dsigmoid(x,e):
    return x*(1-x)*e

def tanh(x):
    return  np.tanh(x)

def dtanh(x,e):
    return  (1-(x**2))*e
    
