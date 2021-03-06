import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

dataset = pd.read_csv('databases/sonar.all-data')
dataset.replace(['M', 'R'], [0, 1], inplace=True)
X = dataset.iloc[:, :60].values
d = dataset.iloc[:, 60:].values

# Embaralhar
x = np.arange(len(d))
np.random.shuffle(x)
X_new = X[x]
d_new = d[x] 


# /Embaralhar
X_base_de_treinamento = X_new[:155,:]
d_base_de_treinamento = d_new[:155,:]
X_base_de_testes = X_new[155:,:]
d_base_de_testes = d_new[155:,:]

p = Perceptron(len(X_base_de_treinamento[0]), epochs=1000)
p.train(X_base_de_treinamento, d_base_de_treinamento)
p.printMatrizparaMatriz(X_base_de_testes,d_base_de_testes)
p.printValoresParaPlanilha()

#plt.xlim(-1,3)
#plt.ylim(-1,3)
#for i in range(len(d)):
 #   if d[i] == 1:
 #       plt.plot(X[i, 0], X[i, 1], 'ro')
 #   else:
#       plt.plot(X[i, 0], X[i, 1], 'bo')
       
#f = lambda x: (p.weights[0]/p.weights[2]) - (p.weights[1]/p.weights[2] * x)
vxH = list(range(-1,3))
#yH = list(map(f, xH))
#plt.plot(xH, yH, 'y-')






#print(p.predict(X[0]))
#print(p.predict(X[1]))
#print(p.predict(X[2]))
#print(p.predict(X[3]))
#
#plt.xlim(-1,3)
#plt.ylim(-1,3)
#for i in range(len(d)):
#    if d[i] == 1:
#        plt.plot(X[i, 0], X[i, 1], 'ro')
#    else:
#        plt.plot(X[i, 0], X[i, 1], 'bo')
        
#f = lambda x: (p.weights[0]/p.weights[2]) - (p.weights[1]/p.weights[2] * x)
#xH = list(range(-1,3))
#yH = list(map(f, xH))
#plt.plot(xH, yH, 'g-')
