import numpy as np
from NeuralPy.ERROR import*
from NeuralPy.Activations import*
###########     LAYER    ###############

class Layer():
    def __init__(self):
        #PARAMS
        self.type = None
        self.Lr = None
        self.active = None
        self.d_active = None
        self.n_params = 0
        self.input_shape = None
        self.output_shape = None

    #ASSIGNING ACTIVATION FUNCTIONS
    def assign_activation(self,a):
        Act = Activations()
        if a not in list(Act.pairs):
            raise NameError(f"activation function ({a}) is not defined")
        return Act.pairs[a]
    
    #FEED FORWARD
    def predict(self,x):
        raise NotImplementedError()

    #BACK PROPAGATION
    def backwards(self,x):
        raise NotImplementedError()

    #GARDIENTS
    def Gard(self,x):
        raise NotImplementedError()



################    DENSE LAYER    ###################



class Dense(Layer):

    def __init__(self,input_shape,n_nodes,activation = 'tanh'):
        super().__init__()
        #PROPERTY
        self.type = "Dense"
        self.active , self.d_active = self.assign_activation(activation)
        self.n_params = (input_shape[1]+1)*n_nodes
        self.input_shape = input_shape
        self.output_shape = (None,n_nodes)

        #WEIGHTS AND BIASES
        a = np.sqrt(6/(input_shape[1]+n_nodes))
        self.weights = np.random.uniform(-a,a,input_shape[1]*n_nodes).reshape((input_shape[1],n_nodes))       
        self.bias = np.random.random((1,n_nodes))*2-1
        self.Vdw = np.zeros_like(self.weights)
        self.Vdb = np.zeros_like(self.bias)
        
    #FEED FORWARD DENSE
    def predict(self,inputs):
        self.input = inputs
        self.output = self.active( np.dot(inputs,self.weights)+ self.bias )
        return self.output.copy()

    #BACK PROPAGATION
    def backwards(self,error,gamma=0.01,beta=0.9):
        dO = self.d_active(self.output,error)
        dE = np.dot(dO,self.weights.T)
        
        dW = np.dot(self.input.T,dO)
        dB = np.expand_dims(np.sum(dO,axis=0),axis=0)
        
        self.Vdw = beta*self.Vdw + dW*(1-beta)
        self.Vdb = beta*self.Vdb + dB*(1-beta)

        self.weights += self.Vdw*gamma
        self.bias += self.Vdb*gamma
        
        return dE

    #GARDIENTS
    def Grad(self,error):
        return np.dot(self.d_active(self.output,error),self.weights.T)



################     CONVOLUTIONAL LAYER     ##################


    
class Conv2D(Layer):
    def __init__(self,n_filters,filter_shape,input_shape,strides=1,padding='none',padder=None,use_bias = True):
        #INPUT SHAPE OF FORM (NONE,IMX,IMY,IMS)
        
        if not padding in ['same','none','valid','custom']:
            raise ConvError(f'padding must have values ["same","none","valid","custom"]')
        #PARAMS
        self.type = 'Conv2D'
        self.s = strides 
        self.nf = n_filters
        self.n_params = np.product(filter_shape)+ n_filters
        self.use_bias = use_bias
        self.fshape = (filter_shape[0],filter_shape[1],input_shape[3],n_filters)

        self.padding = False
        self.padder = ((0,0),(0,0),(0,0),(0,0))
        self.input_shape = input_shape
        self.output_shape = (None,int((input_shape[1]-filter_shape[0])/self.s +1),int((input_shape[2]-filter_shape[1])/self.s +1),self.nf)

        #PADDING AND INPUT-OUTPUT SHAPES
        if not(padding.lower() == 'none' or padding.lower() == 'valid'):
            self.padding = True
            
        if padding.lower() == 'same':
            fx = filter_shape[0]
            fx1 = int((fx-1)/2)
            fx0 = fx-1-fx1

            fy = filter_shape[1]
            fy1 = int((fy-1)/2)
            fy0 = fy-1-fy1
            self.padder = ((0,0),(fx1,fx0),(fy1,fy0),(0,0))
            self.output_shape = input_shape
            
        elif padding.lower() == 'custom':
            self.padder = padder
            px = np.sum(padder[1])
            py = np.sum(padder[2])
            imx,imy = input_shape[1]+px,input_shape[2]+py
            x,y = int((imx-fx)/self.s)+1,int((imy-fy)/self.s)+1
            output_shape = (input_shape[0],x,y,self.nf)

        #INSTANTIATING KERNALS AND KERNAL BIAS
        self.kernals = np.random.random(self.fshape)*2-1#/(self.nf*input_shape[3])
        self.bias = np.random.random((1,n_filters))*2-1#/(self.nf)

        self.Vdk = np.zeros_like(self.kernals)
        self.Vdb = np.zeros_like(self.bias)
        

    #FEED FORWARD (CONVOLVE)
    def predict(self,inputs):
        
        if self.padding:
            inputs = np.pad(inputs,self.padder)
            
        self.input = inputs
        nim,imx,imy,ims = inputs.shape
        fx,fy,fs,nf  = self.fshape
        x,y = int((imx-fx)/self.s)+1,int((imy-fy)/self.s)+1
        c_map = np.zeros((nim,x,y,nf))

        for i in range(x):
            for j in range(y):
                c_map[:,i,j] += np.tensordot(inputs[:,i*self.s:i*self.s+fx,j*self.s:j*self.s+fy],self.kernals,axes=([1,2,3],[0,1,2]))
        return c_map + self.bias if self.use_bias else c_map

    #BACK PROPAGATION
    def backwards(self,e,gamma=0.01,beta=0.9):
        nim,imx,imy,ims = self.input.shape
        fx,fy,fs,nf  = self.fshape
        x,y = int((imx-fx)/self.s)+1,int((imy-fy)/self.s)+1
        de = np.zeros(self.input.shape)
        df = np.zeros(self.fshape)
        fd = (nim)
        ed = (fx*fy*nf)

        for i in range(x):
            for j in range(y):
                df += np.tensordot(self.input[:,i*self.s:i*self.s+fx,j*self.s:j*self.s+fy],e[:,i,j],axes=([0],[0]))/fd
                de[:,i*self.s:i*self.s+fx,j*self.s:j*self.s+fy]+=np.tensordot(e[:,i,j],self.kernals,axes=([1],[3]))/ed

        self.Vdk = self.Vdk*beta + (1-beta)*df
        self.kernals += self.Vdk*gamma
        
        if self.use_bias:
            dB = np.mean(e,axis=(0,1,2))
            self.Vdb = self.Vdb*beta + (1-beta)*dB
            self.bias += self.Vdb*gamma
        return de[:,self.padder[1][0]:imx-self.padder[1][1],self.padder[2][0]:imx-self.padder[2][1]]

    #GRADIENTS
    def Grad(self,e):
        nim,imx,imy,ims = self.input.shape
        fx,fy,fs,nf  = self.fshape
        x,y = int((imx-fx)/self.s)+1,int((imy-fy)/self.s)+1
        de = np.zeros(self.input.shape)
        ed = (fx*fy*nf)

        for i in range(x):
            for j in range(y):
                de[:,i*self.s:i*self.s+fx,j*self.s:j*self.s+fy]+=np.tensordot(e[:,i,j],self.kernals,axes=([1],[3]))/ed
    
        return de[:,self.padder[1][0]:imx-self.padder[1][1],self.padder[2][0]:imx-self.padder[2][1]]


####################    FLATTEN LAYER    ####################


class Flatten(Layer):
    
    def __init__(self,input_shape=None):
        super().__init__()
        #PARAMS
        self.type = "Flatten"
        self.input_shape = input_shape
        self.output_shape = (input_shape[0],np.product(np.array(input_shape)[1:]))
        self.Oshape = list(self.output_shape)
        self.Ishape = list(self.input_shape)

    #FEED FORWARD
    def predict(self,inputs):
        self.Oshape[0] = inputs.shape[0]
        self.Ishape = inputs.shape
        return inputs.reshape(self.Oshape)

    #BACK PROPAGATION
    def backwards(self,errors,gamma=0.01,beta=0.9):
        return errors.reshape(self.Ishape)

    #GRADIENTS
    def Grad(self,errors):
        return errors.reshape(self.Ishape)


###################    ACTIVATION LAYER    ###################


class Activation(Layer):

    def __init__(self,activation,input_shape=None):
        super().__init__()
        #PARAMS
        self.type = 'Activation'
        self.Type = activation
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.output = None
        self.input = None

        #ASSIGNING ACTIVATION
        self.active,self.dactive = self.assign_activation(activation)

    #FEED FORWARD
    def predict(self,inputs):
        self.output = self.active(inputs)
        return self.output

    #BACK PROPAGATION
    def backwards(self,errors,gamma=0.01,beta=0.9):
        return self.dactive(self.output,errors)

    #GRADIENTS
    def Grad(self,errors):
        return self.dactive(self.output,errors)


###################    RECURRENT LAYER    ###################

#apply momentum to wheghts updation
class Recurrent(Layer):      

    def __init__(self,n_nodes,n_inputs,n_hidden,activation=['tanh','tanh']):
        super().__init__()
        self.type = "Recurrent"

        #PARAMS
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        #ACTIVATIONS : [0] = output_activation,[1] = hidden activation
        self.active , self.d_active = self.assign_activation(activation[0])
        self.H_active ,self.H_d_active = self.assign_activation(activation[1])

        #WEIGHTS AND BIASES
        self.Whx = np.random.random((n_inputs,n_hidden))*1.6-0.8
        self.Whh = np.random.random((n_hidden,n_hidden))*1.6-0.8
        self.Why = np.random.random((n_hidden,n_nodes))*1.6-0.8
        self.H_zero = np.random.random((1,n_hidden))*1.6-0.8

            #VELOCITY GRAD
        self.VdW_hx = np.zeros_like(self.Whx)
        self.VdW_hh = np.zeros_like(self.Whh)
        self.VdW_hy = np.zeros_like(self.Why)
        self.VdH = np.zeros_like(self.H_zero)
        
        #BIASES
        self.By = np.random.random((1,n_nodes))*1.6-0.8
        self.Bh = np.random.random((1,n_hidden))*1.6-0.8
        
            #VELOCITY GRAD
        self.Vdby = np.zeros_like(self.By)
        self.Vdbh = np.zeros_like(self.Bh)

    #FEED FORWARD
    def predict(self,inputs):
        self.inputs = inputs
        self.output = []
        self.hiddens = []
        self.inps = []
    
        for i in range(inputs.shape[0]):
            hidden = self.H_zero
            H = [hidden]
            I = []
            O = []
         
            for j in range(inputs[i].shape[0]):
                hidden = self.H_active( np.dot(inputs[i][j].reshape(1,-1) ,self.Whx) + np.dot(hidden ,self.Whh) + self.Bh)
                H.append(hidden)
                I.append(inputs[i][j].reshape(1,-1))
    
            self.output.append(self.active( np.dot(hidden,self.Why) + self.By ).flatten())
            self.inps.append(I)
            self.hiddens.append(H)
            
        return np.array(self.output)

    #BACK PROPAGATION
    def backwards(self,e,gamma=0.1,beta=0.9):
        inputs = self.inputs
        dW_hh = np.zeros_like(self.Whh)
        dW_hx = np.zeros_like(self.Whx)
        dW_hy = np.zeros_like(self.Why)
        dI = []
        
        for i in range(self.inputs.shape[0]):
            dO = self.d_active(self.output[i].reshape(1,-1),e[i].reshape(1,-1))
            dW_hy += np.dot(self.hiddens[i][-1].T,dO)
            self.By += np.expand_dims(np.sum(dO,axis=0)*gamma,axis=0)
            
            dh = np.dot(dO ,self.Why.T )
            dX = np.zeros_like(self.inps[i])
            
            inv_hidden = self.hiddens[i][::-1]
            inv_inputs = self.inps[i][::-1]  
            
            for j in range(self.inputs[i].shape[0]):
                dO = self.H_d_active(inv_hidden[j],dh)
                self.Bh += np.expand_dims(np.sum(dO,axis=0)*gamma,axis=0)
                dW_hh += np.dot(inv_hidden[j+1].T,dO)
                dW_hx += np.dot(inv_inputs[j].T.reshape(-1,1),dO)
                dh = np.dot(dO , self.Whh.T)
                dX[j] += np.dot(dO , self.Whx.T)
                
            dI.append(dX)
            self.H_zero += dh*gamma
            
        self.Whh += dW_hh*gamma
        self.Whx += dW_hx*gamma
        self.Why += dW_hy*gamma

        return np.array(dI)

    #GRADIENTS
    def Grad(self,e):
        dI = []
        
        for i in range(inputs.shape[0]):
            dO = self.d_active(self.output[i].reshape(1,-1),e[i].reshape(1,-1))
            dh = np.dot(dO ,self.Why.T )
            dX = np.zeros_like(self.inps[i])
            
            inv_hidden = self.hiddens[i][::-1]
            inv_inputs = self.inps[i][::-1]  
            
            for j in range(inputs[i].shape[0]):
                dO = self.H_d_active(inv_hidden[j],dh)
                dh = np.dot(dO , self.Whh.T)
                dX[j] += np.dot(dO , self.Whx.T)
                
            dI.append(dX)

        return np.array(dI)


##########################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    r = Recurrent(1,1,5,activation=['sigmoid','tanh'])
    inputs = np.array([[[0],[1],[1]],
                   [[1],[1],[0]],
                   [[0],[1],[0]],
                   [[1],[1],[1]],
                   [[1],[0],[0]],
                   [[0],[0],[0]]
                    ]).astype('float')
    outputs = np.array([[1],[ 0],[0],[0],[1],[0]])
    H = []
    for i in range(1000):
        p = r.predict(inputs)
        e = outputs-p
        r.backwards(e)
        H.append(np.mean(e**2))
    
    
    plt.plot(H)
    plt.show()
