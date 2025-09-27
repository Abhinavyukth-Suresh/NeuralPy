import numpy as np
from NeuralPy.Activation_functions import*
from NeuralPy.Layers import*
from NeuralPy.errors import *
from NeuralPy.ERROR import*

###########     LAYER    ###############
  
class Neural_Network():
    
    def __init__(self,model=None):
        #model= [layer1(params),layer2(params)....layerN(params)]
        self.model = []
        self.optimizers = ['SGD','adagrad','VanillaGD']
        self.optimizer = 'Vanilla'
        self.Error = None
        self.compiled = False
    
        if not model is None:
            self.model = model

    #ADD LAYERS
    def add(self,layer,params=None):
        if not issubclass(type(layer),Layer):
            raise LayerError(f"you CANNOT add a Non-NNLayer to model")

        self.model.append(x)
        #self.Summary.append((x.type,x.input_shape,x.output_shape,x.n_params))
        
    #FEED FORWARD
    def predict(self,inputs):
        for i in range(len(self.model)):
            inputs = self.model[i].predict(inputs)
        return inputs

    #BACK PROPAGATION
    def backwards(self,error,gamma=0.01,beta=0.9):
        model = self.model[::-1]
        for i in range(len(model)):
            error = model[i].backwards(error,gamma,beta)
        return error

    #GRADIENTS
    def Gradients(self,error):
        model = self.model[::-1]
        for i in range(len(model)):
            error = model[i].Grad(error)
        return error

    #COMPILING THE MODEL
    def compile(self,error='MSE',optimizer='VanillaGD',from_logits = False):
        self.Compile = (error,optimizer,from_logits)
        self.dError,self.Error = Errors().Errors[error]
        self.compiled = True

    #TRAINING THE MODEL
    def fit(self,inputs,outputs,epochs=500,gamma=0.007,n_batches = 1,
            history=False,beta=0.9):
        H = []
        #if not self.compiled:
        #    raise CompileError("MODEL NOT COMPILED!!!")

        input_batches = np.array_split(inputs,n_batches)
        output_batches = np.array_split(outputs,n_batches)
        
        model = self.model[::-1]
        
        for e in range(epochs):
            for i in range(n_batches):
                pred = self.predict(input_batches[i])
                error = output_batches[i]-pred
                self.backwards(error,gamma,beta)
            H.append(np.mean(error**2)) if history else 0
      
        return H

    #TRAINING THE MODEL
    def dfit(self,inputs,outputs,epochs=500,gamma=0.007,n_batches = 1,
            history=False,beta=0.9,decay=False,decay_rate = None,decay_multiplier=0.5):
        H = []
        #if not self.compiled:
        #    raise CompileError("MODEL NOT COMPILED!!!")

        input_batches = np.array_split(inputs,n_batches)
        output_batches = np.array_split(outputs,n_batches)
        
        model = self.model[::-1]

        if decay_rate is None:
            decay_rate = gamma*decay_multiplier/epochs

        for e in range(epochs):
            
            gamma *= 1/(1+decay_rate*e) if decay else 1
            for i in range(n_batches):
                pred = self.predict(input_batches[i])
                error = output_batches[i]-pred
                self.backwards(error,gamma,beta)
            H.append(np.mean(error**2)) if history else 0
      
        return H
    
    #SUMMARIZING THE MODEL
    @property
    def Summary(self):
        summ = []
        for x in self.model:
            summ.append((x.type,x.input_shape,x.output_shape,x.n_params))
        return summ
    
    @property
    def summary(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        clmn  = [" Layer "," input_shape "," output_shape "," n_params "]
        table.field_names = clmn
        table.add_rows(self.Summary)
        print(table)
        total_params = 0
        for i in self.Summary:
            total_params += i[3]
        print(f"Total parameters :{total_params}")
        print(f"Compilation satus :{['Error : ' +str(self.Compile[0]),'Optimizer : '+str(self.Compile[1])] if self.compiled else 'Not compiled!!'}")

    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    inp = np.random.random((10,8,8,3))
    out = np.random.random((10,1))

    nn = Neural_Network([
    Conv2D(32,(3,3),(None,8,8,3)),
    Conv2D(16,(3,3),(None,6,6,32)),
    Conv2D(8,(3,3),(None,4,4,16)),
    Flatten((None,2,2,8)),
    Activation('tanh'),
    Dense((None,32),3,'tanh'),
    Dense((None,3),3,'tanh'),
    Dense((None,3),1,'tanh')
    ])

    
    nn.compile()

    o = nn.fit(inp,out,history=True)
    
    plt.plot(o,color='red')
    plt.show()
