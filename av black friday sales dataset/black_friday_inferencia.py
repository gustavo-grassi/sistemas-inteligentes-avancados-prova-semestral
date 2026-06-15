import pickle
import pandas as pd
import numpy as np

colunas_treino = pickle.load(open('estrutura_atributos_bf.pkl', 'rb'))

modelo_categoria = pickle.load(open('modelo_bf_product_category.pkl', 'rb'))
modelo_pagamento = pickle.load(open('modelo_bf_payment_method.pkl', 'rb'))
modelo_faixa_etaria = pickle.load(open('modelo_bf_age_group.pkl', 'rb'))

nova_venda_orig = pd.DataFrame([['Female', 'New York', 'Premium']], 
                               columns=['gender', 'city', 'customer_segment'])

nova_venda_processada = pd.get_dummies(nova_venda_orig, dtype=int)
final_input = pd.DataFrame(columns=colunas_treino)
final_input = pd.concat([final_input, nova_venda_processada]).fillna(0)
final_input = final_input[colunas_treino]

print("SISTEMA INTELIGENTE BLACK FRIDAY: INFERÊNCIA")

pred_cat = modelo_categoria.predict(final_input)[0]
prob_cat = np.max(modelo_categoria.predict_proba(final_input)) * 100

pred_pag = modelo_pagamento.predict(final_input)[0]
prob_pag = np.max(modelo_pagamento.predict_proba(final_input)) * 100

pred_age = modelo_faixa_etaria.predict(final_input)[0]
prob_age = np.max(modelo_faixa_etaria.predict_proba(final_input)) * 100

print(f"\nCategoria do produto indicada: {pred_cat} (Certeza: {prob_cat:.2f}%)")
print(f"Forma de pagamento provável: {pred_pag} (Certeza: {prob_pag:.2f}%)")
print(f"Faixa etária provável do comprador: {pred_age} (Certeza: {prob_age:.2f}%)")