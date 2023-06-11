import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import KFold




def remove_outliers(data, threshold=1.5):
    
    # Calcula o primeiro quartil (Q1) e o terceiro quartil (Q3) dos dados
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    # Calcula o intervalo interquartil (IQR)
    iqr = q3 - q1
    
    # Calcula os limites inferior e superior para determinar os outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Remove os outliers dos dados
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data


def NormalizarDiferenca(ativo_diferenca):

    ativo_array = np.array(ativo_diferenca["Diferenca"]).reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(ativo_array)

    scaled_data_list = pd.DataFrame(scaled_data.flatten().tolist(),columns=["Diferenca"])

    return scaled_data_list


def preparar_dados_para_treinamento(anteriores, base_treinamento_normalizada):

    previsores = []
    preco_real = []

    for i in range(anteriores, len(base_treinamento_normalizada)):
        
        
        previsores.append(base_treinamento_normalizada[i-anteriores:i])
        preco_real.append(base_treinamento_normalizada[i])

    previsores, preco_real = np.array(previsores), np.array(preco_real)
    previsores = np.reshape(
        previsores, (previsores.shape[0], previsores.shape[1], 1))

    return previsores, preco_real




def criarRedeNeural(previsores, preco_real, filepath, epocas=300, validacao_cruzada=0, ativacao="linear", otimizador="adam", minimo_delta=1e-15, paciencia_es=10, batch=40):

    regressor = Sequential()

    # 1º
    regressor.add(LSTM(units=70, return_sequences=True,
                  input_shape=(previsores.shape[1], 1)))
    regressor.add(Dropout(0.3))

    # 2º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))

    # 3º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))

    # 4º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))

    # 5º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))
    
    # 6º
    regressor.add(LSTM(units=70))
    regressor.add(Dropout(0.3))

    # 7º
    regressor.add(Dense(units=1, activation=ativacao))

    regressor.compile(optimizer=otimizador, loss='mean_squared_error', metrics=[
                      'mean_absolute_error'])

    es = EarlyStopping(monitor="loss", min_delta=minimo_delta,
                       patience=paciencia_es, verbose=1)
    rlr = ReduceLROnPlateau(monitor="loss", factor=0.06, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath=filepath, monitor="loss",
                          save_best_only=True, verbose=1)

    if validacao_cruzada == 1:

        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(previsores):
            X_train, X_test = previsores[train_index], previsores[test_index]
            y_train, y_test = preco_real[train_index], preco_real[test_index]

            regressor.fit(X_train, y_train, epochs=epocas,
                          batch_size=batch, callbacks=[es, mcp])
            score = regressor.evaluate(X_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    else:

        regressor.fit(previsores, preco_real, epochs=epocas,
                      batch_size=batch, callbacks=[es, mcp])

    return regressor





ativo ="PETR3.SA"

ativo_df = yf.download(tickers = ativo,period = "1y", start="2019-01-01")

ativo_df["Diferenca"] = ativo_df["Close"] - ativo_df["Open"]

ativo_df_semOutliers = pd.DataFrame(remove_outliers(ativo_df["Diferenca"]),columns=["Diferenca"])

diferenca_normalizada = NormalizarDiferenca(ativo_df_semOutliers)

previsorees,resultado = preparar_dados_para_treinamento(15, diferenca_normalizada["Diferenca"])





















