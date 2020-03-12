import numpy as np
from activation_functions import heaviside_step_function

class Perceptron():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=1000, learning_rate=2.5e-3):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        
        
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return self.act_func(u)
        
    
    def train(self, training_inputs, labels):
        self.initial_weights = self.weights
        error = True
        for e in range(self.epochs):
            error = False
            print(f'------------------ epoch {e +1}')
            print(f'weights {self.weights}')
            for inputs, label in zip(training_inputs, labels):
                predicton = self.predict(inputs)
                if predicton != label:
                    inputs = np.append(-1, inputs)
                    self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
                    error = True
                    break
                else:
                    print(f'Everything is OK!')
            
            print('')
            if not error:
                break
        self.final_weights = self.weights
        
    def printValoresParaPlanilha(self):
        print(f'Pesos Iniciais: {self.initial_weights}')
        print(f'Pesos Finais: {self.final_weights}')
        print(f'RFalso: {self.RFalso}')
        print(f'MFalso: {self.MFalso}')
        print(f'RPositivo: {self.RPositivo}')
        print(f'MPositivo: {self.MPositivo}')
    
    def printMatrizparaMatriz(self, training_inputs, labels):        
        self.RPositivo = 0
        self.RFalso = 0
        self.MPositivo = 0
        self.MFalso = 0 
        
        for inputs, label in zip(training_inputs, labels):
            predicton = self.predict(inputs)
            #dataset.replace(['M', 'R'], [0, 1], inplace=True)
            if predicton != label:
                if label == 1:
                    self.RFalso += 1 
                else:
                    self.MFalso += 1
            else:
                if label == 1:
                    self.RPositivo += 1 
                else:
                    self.MPositivo += 1
                    