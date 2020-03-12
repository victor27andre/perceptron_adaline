import numpy as np
from activation_functions import signum_function

class Adaline():

    def __init__(self, input_size, act_func=signum_function, epochs=1000, learning_rate=2.5e-3, precision=1e-6):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1)
        self.precision = precision
        self.input_size = input_size + 1

    def printMatrizparaMatriz(self, training_inputs, labels):        
        self.RPositivo = 0
        self.RFalso = 0
        self.MPositivo = 0
        self.MFalso = 0 
        
        for inputs, label in zip(training_inputs, labels):
            predicton = self.act_func(self.predict(inputs))
            
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
            
    def predict (self,inputs): 
        inputs = np.append(-1,inputs)
        u = np.dot(inputs, self.weights)
        return u
    
    def printValoresParaPlanilha(self):
        
        print(f'Pesos Iniciais: {self.initial_weights}')
        print(f'Pesos Finais: {self.final_weights}')
        print(f'RFalso: {self.RFalso}')
        print(f'MFalso: {self.MFalso}')
        print(f'RPositivo: {self.RPositivo}')
        print(f'MPositivo: {self.MPositivo}')
                
    
    def train(self, training_inputs, labels):
        self.initial_weights = self.weights
        for e in range(self.epochs):
            print(f'----------------- epoch {e +1}')
            eqmPrevious = self.eqm(training_inputs, labels)
            print(f'eqmPrevious: {eqmPrevious}')
            for inputs, label in zip(training_inputs, labels):
                predicton = self.predict(inputs)
                #Adds
                inputs = np.append(-1,inputs)
                self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
            eqmActual = self.eqm(training_inputs,labels)
            print(f'eqmActual: {eqmActual}')
            print('')
            if self.precision >= abs(eqmActual - eqmPrevious):
                break
        self.final_weights = self.weights
        return e + 1
    
    def eqm(self, trainInputs, trainOutputs):
        eqmCalc = 0
        for inputs, label in zip(trainInputs, trainOutputs):
            prediction = self.predict(inputs)
            eqmCalc = (label - prediction) ** 2
        eqmCalc += eqmCalc / self.input_size
        return eqmCalc
    
  
    
