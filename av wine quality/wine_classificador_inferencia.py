import pickle
import pandas as pd

modelo_vinho = pickle.load(open('melhor_classificador_vinho.pkl', 'rb'))

colunas_atributos = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

novo_vinho = pd.DataFrame([[7.4, 0.34, 0.30, 2.2, 0.075, 17.0, 44.0, 0.9968, 3.24, 0.58, 10.5]], 
                          columns=colunas_atributos)

predicao_qualidade = modelo_vinho.predict(novo_vinho)

print('SISTEMA DE INFERÊNCIA DE QUALIDADE DE VINHOS')
print(f'A qualidade estimada (nota) para o vinho enviado é: {predicao_qualidade[0]}')