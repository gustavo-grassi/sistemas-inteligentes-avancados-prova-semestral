# Importar libs
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import math
import numpy as np

dados = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',')

dados_atributos = dados.drop(columns=['time', 'DEATH_EVENT'])

colunas_num = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
colunas_bin = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

dados_num = dados_atributos[colunas_num]
dados_bin = dados_atributos[colunas_bin]

scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

pickle.dump(normalizador, open('normalizador_heart.pkl', 'wb'))

dados_num_norm = normalizador.transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

dados_bin_norm = pd.get_dummies(dados_bin, columns=colunas_bin, prefix_sep='_', dtype=int)

dados_norm = dados_num_norm.join(dados_bin_norm, how='left')
print("### Primeiras linhas dos dados normalizados ###")
print(dados_norm.head(10))

distorcoes = []
K = range(1, 21)
for i in K:
    cluster_heart = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_norm)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_norm, cluster_heart.cluster_centers_, 'euclidean'), axis=1
            ) / dados_norm.shape[0]
        )
    )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('\nNumero otimo de clusters =', numero_clusters_otimo)

# Treinar e salvar o modelo final de clusters
cluster_heart = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10).fit(dados_norm)
pickle.dump(cluster_heart, open('cluster_heart.pkl', 'wb'))
