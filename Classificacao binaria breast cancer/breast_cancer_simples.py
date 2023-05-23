# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:27:50 2022

@author: Harrison Hakanen
"""


import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import confusion_matrix,accuracy_score



previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

#SPLIT DO DATABASE
previsores_treinamento, previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size=0.25)


classificador = Sequential()

#A formula para definir a quantidade de neuronios na camada ocuta foi
#total de atribtos + total de classes / 2 e arredonda para cima
classificador.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform',input_dim=30))

classificador.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))

classificador.add(Dense(units=1,activation='sigmoid'))


#A loss function foi determinada pela binary_crossentropy pois so temos 
#duas classes como resultado, ou seja um problema binario

#----------------CRIANDO O OTIMIZADOR----------------
#E criado essa variavel para poder alterar o decay do otimizador
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.0001)

otimizador = keras.optimizers.Adam(learning_rate=lr_schedule,clipvalue=0.4)
#----------------------------------------------------


classificador.compile(optimizer=otimizador, loss="binary_crossentropy",metrics=["binary_accuracy"])

classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10,epochs=100)


#AVALIANDO A ACURACIA DO MODELO
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5)

#Medicao de forma manual
precisao = accuracy_score(classe_teste,previsoes)
matrix = confusion_matrix(classe_teste,previsoes)


#Medicao automatica

resultado = classificador.evaluate(previsores_teste,classe_teste)



#Visualização de pesos
pesos_camada_0 = classificador.layers[0].get_weights
pesos_camada_1 = classificador.layers[1].get_weights
pesos_camada_2 = classificador.layers[2].get_weights



