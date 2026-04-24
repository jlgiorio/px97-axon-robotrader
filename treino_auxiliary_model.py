import pandas as pd
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.metrics import precision_recall_curve
import joblib

# Defina as colunas que devem ser usadas (presentes em todos os arquivos)
colunas_usadas = [
    'timestamp', 'ativo', 'contratos', 'proba', 'rsi',
    'preco_entrada', 'alvo', 'stop', 'preco_saida', 'lucro', 'resultado', 'pontos'
]

# Lista todos os arquivos de operações no padrão especificado
arquivos = glob.glob('dados/px97_axon_operacoes_log_*.csv')

# Lê e concatena apenas as colunas desejadas de todos os arquivos
dfs = []
for arq in arquivos:
    df = pd.read_csv(arq, usecols=lambda c: c in colunas_usadas)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

# Exibe o DataFrame resultante para análise
print(df_total)
print(f"\nTotal de operações carregadas: {len(df_total)}")
print(f"Colunas presentes: {list(df_total.columns)}")

# 1. Remove operações TIME_DROP e mantém apenas GAIN/LOSS
df_total = df_total[df_total['resultado'].isin(['GAIN', 'LOSS'])]

# 2. Remove apenas linhas com NaN nas features (não em todas as colunas)
df_total = df_total.dropna(subset=colunas_usadas)

print("Após filtrar GAIN/LOSS:", len(df_total))
print("Após remover NaN nas features:", len(df_total))

# 3. Defina as features que serão usadas
features = [
    'proba', 'rsi', 'preco_entrada', 'alvo', 'stop'
]
X = df_total[features]
y = df_total['resultado'].map({'GAIN': 1, 'LOSS': 0})

# 4. Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Treina o modelo RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Avaliação do modelo
y_pred = clf.predict(X_test)
print("=== Relatório de Classificação (Validação) ===")
print(classification_report(y_test, y_pred))

# Após treinar o modelo e antes de salvar:
y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilidade de GAIN

# Encontra o melhor threshold para maximizar o F1-score
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Threshold ótimo sugerido pelo histórico: {best_threshold:.2f}")

import matplotlib.pyplot as plt

# Visualizar a importância das features
importances = clf.feature_importances_
feature_names = features

# Exibe as importâncias no terminal
for name, imp in zip(feature_names, importances):
    print(f"Feature: {name:15s} | Importância: {imp:.4f}")

# Gráfico de barras das importâncias
plt.figure(figsize=(8, 4))
plt.bar(feature_names, importances)
plt.title("Importância das Features - RandomForest")
plt.ylabel("Importância")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()

# 7. Salva o modelo treinado
import os
# Salva modelo e threshold juntos
os.makedirs('modelos/auxiliar', exist_ok=True)
joblib.dump({'modelo': clf, 'threshold': best_threshold}, 'modelos/auxiliar/modelo_auxiliar_operacoes_SemATR.joblib')
print("Modelo auxiliar salvo em: modelos/auxiliar/modelo_auxiliar_operacoes_SemATR.joblib")