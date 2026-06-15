import pickle
import pandas as pd

estrutura_colunas = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium',
    'anaemia_0', 'anaemia_1', 'diabetes_0', 'diabetes_1', 'high_blood_pressure_0', 'high_blood_pressure_1',
    'sex_0', 'sex_1', 'smoking_0', 'smoking_1'
]
paciente_normalizado_df = pd.DataFrame(columns=estrutura_colunas)

novo_paciente_num = pd.DataFrame([[65.0, 250.0, 35.0, 220000.0, 1.3, 135.0]], 
                                 columns=['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium'])

novo_paciente_bin = pd.DataFrame([[0, 1, 1, 0, 0, 1, 0, 1, 1, 0]], 
                                 columns=[
                                     'anaemia_0', 'anaemia_1', 
                                     'diabetes_0', 'diabetes_1', 
                                     'high_blood_pressure_0', 'high_blood_pressure_1',
                                     'sex_0', 'sex_1', 
                                     'smoking_0', 'smoking_1'
                                 ])

normalizador = pickle.load(open('normalizador_heart.pkl', 'rb'))
novo_paciente_num_norm = normalizador.transform(novo_paciente_num)
novo_paciente_num_norm = pd.DataFrame(novo_paciente_num_norm, columns=novo_paciente_num.columns)

novo_paciente_completo = novo_paciente_num_norm.join(novo_paciente_bin)

final_input = pd.concat([novo_paciente_completo, paciente_normalizado_df]).fillna(0)
final_input = final_input[estrutura_colunas]

cluster_heart = pickle.load(open('cluster_heart.pkl', 'rb'))
grupo_paciente = cluster_heart.predict(final_input)

print('RESULTADO DA INFERÊNCIA')
print('O paciente desconhecido pertence ao grupo clínico (Cluster):', grupo_paciente[0])