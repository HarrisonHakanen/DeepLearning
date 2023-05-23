# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:26:24 2023

@author: Harrison
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Forma antiga de importar o KerasClassifier mas que ainda é compatível com o cross_val_score do sklearn.model_selection
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 

#from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score



previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

def CriarRede(optimizer,loss_,kernel_initializer_,activation,neurons):
    
    classificador = Sequential()

    classificador.add(Dense(units=neurons,activation=activation,kernel_initializer=kernel_initializer_,input_dim=30))

    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=neurons,activation=activation,kernel_initializer=kernel_initializer_))
    
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units=1,activation='sigmoid'))


    classificador.compile(optimizer=optimizer, loss=loss_,metrics=["binary_accuracy"])
    
    return classificador


classificador = KerasClassifier(build_fn=CriarRede)


parametros={'batch_size':[10,30],
            'epochs':[50,100],
            'optmizer':['adam','sgd'],
            'loss':['binary_cross_entropy','hinge'],
            'kernel_initializer':['random_uniform','normal'],
            'activation':['relu','tanh'],
            'neurons':[16,8]}


grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           scoring="accuracy",
                           cv=5)


grid_search = grid_search.fit(previsores,classe)


melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_