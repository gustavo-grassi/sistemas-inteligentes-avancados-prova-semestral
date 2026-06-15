import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

dados = pd.concat([red_wine, white_wine], ignore_index=True)

dados_atributos = dados.drop(columns=['quality'])
dados_classe = dados['quality']

print('Frequência das classes original:', Counter(dados_classe))

resampler = SMOTE(k_neighbors=2) 
atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)

print('Frequência após o balanceamento (SMOTE):', Counter(classes_b))

X_train, X_test, y_train, y_test = train_test_split(atributos_b, classes_b, test_size=0.3, random_state=42)

modelo_tree = DecisionTreeClassifier(random_state=42)
modelo_rf = RandomForestClassifier(random_state=42)
modelo_knn = KNeighborsClassifier()

print("\nTreinando os Modelos")
modelo_tree.fit(X_train, y_train)
modelo_rf.fit(X_train, y_train)
modelo_knn.fit(X_train, y_train)

pred_tree = modelo_tree.predict(X_test)
pred_rf = modelo_rf.predict(X_test)
pred_knn = modelo_knn.predict(X_test)

acc_tree = accuracy_score(y_test, pred_tree)
acc_rf = accuracy_score(y_test, pred_rf)
acc_knn = accuracy_score(y_test, pred_knn)

print(f"Acurácia Árvore de Decisão: {acc_tree:.4f}")
print(f"Acurácia Random Forest: {acc_rf:.4f}")
print(f"Acurácia KNN: {acc_knn:.4f}")

print("\nRELATÓRIO DO MELHOR MODELO (Random Forest)")
print("Matriz de Confusão:\n", confusion_matrix(y_test, pred_rf))
print("\nMétricas por Classe (Acurácia/Precision/Recall/F1-Score):\n", classification_report(y_test, pred_rf))

pickle.dump(modelo_rf, open('melhor_classificador_vinho.pkl', 'wb'))