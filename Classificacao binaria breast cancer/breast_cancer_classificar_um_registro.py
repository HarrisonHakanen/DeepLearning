# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:49:30 2023

@author: Harrison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense, Dropout

#Forma antiga de importar o KerasClassifier mas que ainda é compatível com o cross_val_score do sklearn.model_selection
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 

#from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score

previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")



def CriarRede():
    
    classificador = Sequential()

    classificador.add(Dense(units=8,activation="relu",kernel_initializer="normal",input_dim=30))

    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=8,activation="relu",kernel_initializer="normal"))
    
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units=1,activation='sigmoid'))
    
    
    #----------------CRIANDO O OTIMIZADOR----------------
    #E criado essa variavel para poder alterar o decay do otimizador
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.0001)

    otimizador = keras.optimizers.Adam(learning_rate=lr_schedule,clipvalue=0.4)
    #----------------------------------------------------

 
    classificador.compile(optimizer=otimizador, loss="binary_crossentropy",metrics=["binary_accuracy"])
    
    return classificador


classificador = CriarRede()

classificador.fit(previsores,classe,batch_size=10,epochs=100)


novo_registro = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])


previsao = classificador.predict(novo_registro)


#Salva o modelo
classificador_json = classificador.to_json()
with open('classificador_breast_cancer.json','w') as json_file:
    json_file.write(classificador_json)
    
#Salva os pesos da rede neural
classificador.save_weights("classificador_breas_cancer_pesos.h5")



#Carregar o modelo
classificador_estrutura = open("classificador_breast_cancer.json","r").read()

classificador = model_from_json(classificador_estrutura)
classificador.load_weights("classificador_breas_cancer_pesos.h5")
classificador.compile(optimizer="adam", loss="binary_crossentropy",metrics=["binary_accuracy"])

previsao_nova = classificador.predict(novo_registro)
