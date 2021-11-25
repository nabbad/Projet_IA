#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:55:40 2021

@author: nourreddineabbad
"""

import numpy as np

x_entrer = np.array(([3., 1.5], [2., 1.], [4., 1.5], [3., 1], [1, 1.5],\
                   [2., 0.5], [5.5, 1.], [1., 1.], [4.5, 1.]), dtype=float)

y = np.array(([1.], [0.], [1.], [0.], [1.], [0.], [1.], [0.],), dtype=float) #donnée de sortie 1 = Rouge / 0 = Bleu

x_entrer = x_entrer/np.amax(x_entrer, axis=0) #normaliser

print(x_entrer)

X = np.split(x_entrer, [8])[0]  #on enlève la dernière valeur 
xPrediction = np.split(x_entrer, [8])[1]


class Neural_Network():
    
    def __init__(self):
        
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        #Neurone Synapse
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #matrice2x3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)#matrice3x1
        
    def forward(self, X):
        """
        Multiplier les valeurs par les poids et calculer la sigmoide
        """
        
        self.z = np.dot(X, self.W1)
        
        self.z2 = self.sigmoid(self.z)
         
        self.z3 = np.dot(self.z2, self.W2)
        
        o = self.sigmoid(self.z3)
        
        return o 
    
    #definition algorithme du gradient
    
    def sigmoid(self, s):
        return 1./(1. + np.exp(-s))
    
    def sigmoidPrime(self, s):
        return s * (1-s) 
    #foction de retropropagation
    
    def backward(self, X, y, o):
        
        self.o_error = y - o #erreur sortie moins entrée
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
    #fonction train
    
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o) 
        
    def predict(self):
        print("Donnée prédite après entrainement : ")
        print("Entréen : \n" + str(xPrediction))
        
        print("Sortie : \n" + str(self.forward(xPrediction)))
        
        if(self.forward(xPrediction) < 0.5):
            print("La fleur est BLEU ! \n")
        else :
            print("La fleur est ROUGE ! \n")
        
    
        
    
NN = Neural_Network()

for i in range(300):
    
    print("#" + str(i) + "\n")
    print("Valeurs d'entrée : \n" + str(X))
    print("Sortie Actuelle : \n" + str(y))
    print("Sortie prédite : \n" + str(np.matrix.round(NN.forward(X), 2)))
    print("\n")
    NN.train(X, y)
    
NN.predict()

#Ceci est un test de Git
#ajout d'un nouveau commentaire 


#ajout d'un text sur la branche ajouter texte