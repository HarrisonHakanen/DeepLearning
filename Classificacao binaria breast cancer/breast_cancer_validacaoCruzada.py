# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:00:15 2023

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

from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score



previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

def CriarRede():
    
    classificador = Sequential()

    classificador.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform',input_dim=30))

    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))
    
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units=1,activation='sigmoid'))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.0001)

    otimizador = keras.optimizers.Adam(learning_rate=lr_schedule,clipvalue=0.4)

    classificador.compile(optimizer=otimizador, loss="binary_crossentropy",metrics=["binary_accuracy"])
    
    return classificador



classificador = KerasClassifier(build_fn = CriarRede,epochs=100,batch_size=10)

kfold = StratifiedKFold(n_splits=10,shuffle=True)


resultados = cross_val_score(estimator = classificador,
                             X=previsores,
                             y=classe,
                             cv=kfold,
                             scoring="accuracy")


media = resultados.mean()
desvio = resultados.std()









