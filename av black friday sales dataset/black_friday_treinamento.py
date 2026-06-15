import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

dados = pd.read_csv('retail_black_friday_sales_100k.csv', sep=',')

alvos = ['product_category', 'payment_method', 'age_group']

remover_colunas = ['transaction_id', 'customer_id', 'product_id', 'purchase_date', 'original_price', 'discount_amount', 'final_price'] + alvos
colunas_contexto = [col for col in dados.columns if col not in remover_colunas]

X = pd.get_dummies(dados[colunas_contexto], drop_first=True, dtype=int)

pickle.dump(X.columns.tolist(), open('estrutura_atributos_bf.pkl', 'wb'))

print(f"Atributos de contexto processados. Total de dimensões: {X.shape[1]}")

def calcular_metricas(y_true, y_pred, label):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    sensibilidades = []
    especificidades = []
    
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        espe = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensibilidades.append(sens)
        especificidades.append(espe)
        
    print(f"\nMÉTRICAS PARA AVANÇADA: {label}")
    print(f"Acurácia global: {acc:.4f}")
    print(f"Sensibilidade (Média): {np.mean(sensibilidades):.4f}")
    print(f"Especificidade (Média): {np.mean(especificidades):.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print("Matriz de confusão:\n", cm)

for alvo in alvos:
    print(f"\nIniciando treinamento para o alvo: {alvo}...")
    y = dados[alvo]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    modelo = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    
    predicoes = modelo.predict(X_test)
    
    calcular_metricas(y_test, predicoes, alvo)
    
    pickle.dump(modelo, open(f'modelo_bf_{alvo}.pkl', 'wb'))
