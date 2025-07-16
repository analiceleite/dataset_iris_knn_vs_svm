# Classificação do Dataset Iris - KNN vs SVM com Validação Cruzada
# Implementação completa com métricas de avaliação e comparação

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuração para visualização
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("CLASSIFICAÇÃO DO DATASET IRIS - KNN vs SVM")
print("Validação Cruzada e Análise Comparativa")
print("="*80)

# ==================================================================================
# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ==================================================================================

# Carregar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("\n1. INFORMAÇÕES DO DATASET")
print("-" * 50)
print(f"Shape dos dados: {X.shape}")
print(f"Número de classes: {len(np.unique(y))}")
print(f"Classes: {target_names}")
print(f"Features: {feature_names}")

# Criar DataFrame para melhor visualização
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = [target_names[i] for i in y]

print(f"\nDistribuição das classes:")
print(df['species'].value_counts())

print(f"\nEstatísticas descritivas:")
print(df.describe())

# ==================================================================================
# 2. PRÉ-PROCESSAMENTO DOS DADOS
# ==================================================================================

print("\n2. PRÉ-PROCESSAMENTO")
print("-" * 50)

# Padronização dos dados (importante para SVM e KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dados padronizados com StandardScaler")
print(f"Média após padronização: {np.mean(X_scaled, axis=0)}")
print(f"Desvio padrão após padronização: {np.std(X_scaled, axis=0)}")

# ==================================================================================
# 3. CONFIGURAÇÃO DA VALIDAÇÃO CRUZADA
# ==================================================================================

print("\n3. CONFIGURAÇÃO DA VALIDAÇÃO CRUZADA")
print("-" * 50)

# Configurar validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Validação cruzada estratificada com {cv.n_splits} folds")

# Métricas para avaliação
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# ==================================================================================
# 4. IMPLEMENTAÇÃO E AVALIAÇÃO DO KNN
# ==================================================================================

print("\n4. CLASSIFICADOR K-NEAREST NEIGHBORS (KNN)")
print("-" * 50)

# Testar diferentes valores de K
k_values = range(1, 21)
knn_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
    knn_scores.append(scores.mean())

# Encontrar melhor K
best_k = k_values[np.argmax(knn_scores)]
print(f"Melhor valor de K: {best_k} (acurácia: {max(knn_scores):.4f})")

# Treinar KNN com melhor K
knn_best = KNeighborsClassifier(n_neighbors=best_k)

# Validação cruzada completa para KNN
knn_cv_results = cross_validate(knn_best, X_scaled, y, cv=cv, scoring=scoring)

print(f"\nResultados da Validação Cruzada - KNN (K={best_k}):")
print(f"Acurácia: {knn_cv_results['test_accuracy'].mean():.4f} (±{knn_cv_results['test_accuracy'].std():.4f})")
print(f"Precisão: {knn_cv_results['test_precision_macro'].mean():.4f} (±{knn_cv_results['test_precision_macro'].std():.4f})")
print(f"Revocação: {knn_cv_results['test_recall_macro'].mean():.4f} (±{knn_cv_results['test_recall_macro'].std():.4f})")
print(f"F1-Score: {knn_cv_results['test_f1_macro'].mean():.4f} (±{knn_cv_results['test_f1_macro'].std():.4f})")

# ==================================================================================
# 5. IMPLEMENTAÇÃO E AVALIAÇÃO DO SVM
# ==================================================================================

print("\n5. CLASSIFICADOR SUPPORT VECTOR MACHINE (SVM)")
print("-" * 50)

# Testar diferentes kernels
kernels = ['linear', 'rbf', 'poly']
svm_results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
    svm_results[kernel] = scores.mean()
    print(f"SVM {kernel}: {scores.mean():.4f} (±{scores.std():.4f})")

# Encontrar melhor kernel
best_kernel = max(svm_results, key=svm_results.get)
print(f"\nMelhor kernel: {best_kernel} (acurácia: {svm_results[best_kernel]:.4f})")

# Treinar SVM com melhor kernel
svm_best = SVC(kernel=best_kernel, random_state=42)

# Validação cruzada completa para SVM
svm_cv_results = cross_validate(svm_best, X_scaled, y, cv=cv, scoring=scoring)

print(f"\nResultados da Validação Cruzada - SVM ({best_kernel}):")
print(f"Acurácia: {svm_cv_results['test_accuracy'].mean():.4f} (±{svm_cv_results['test_accuracy'].std():.4f})")
print(f"Precisão: {svm_cv_results['test_precision_macro'].mean():.4f} (±{svm_cv_results['test_precision_macro'].std():.4f})")
print(f"Revocação: {svm_cv_results['test_recall_macro'].mean():.4f} (±{svm_cv_results['test_recall_macro'].std():.4f})")
print(f"F1-Score: {svm_cv_results['test_f1_macro'].mean():.4f} (±{svm_cv_results['test_f1_macro'].std():.4f})")

# ==================================================================================
# 6. MATRIZ DE CONFUSÃO E MÉTRICAS DETALHADAS
# ==================================================================================

print("\n6. MATRIZES DE CONFUSÃO E MÉTRICAS DETALHADAS")
print("-" * 50)

# Dividir dados para teste final
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelos finais
knn_final = KNeighborsClassifier(n_neighbors=best_k)
svm_final = SVC(kernel=best_kernel, random_state=42)

knn_final.fit(X_train, y_train)
svm_final.fit(X_train, y_train)

# Predições
y_pred_knn = knn_final.predict(X_test)
y_pred_svm = svm_final.predict(X_test)

# Função para calcular métricas
def calculate_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n{model_name} - Métricas no Conjunto de Teste:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Revocação: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# Calcular métricas
knn_metrics = calculate_metrics(y_test, y_pred_knn, "KNN")
svm_metrics = calculate_metrics(y_test, y_pred_svm, "SVM")

# Relatórios de classificação detalhados
print(f"\n{'-'*30} RELATÓRIO KNN {'-'*30}")
print(classification_report(y_test, y_pred_knn, target_names=target_names))

print(f"\n{'-'*30} RELATÓRIO SVM {'-'*30}")
print(classification_report(y_test, y_pred_svm, target_names=target_names))

# ==================================================================================
# 7. VISUALIZAÇÕES
# ==================================================================================

print("\n7. GERANDO VISUALIZAÇÕES")
print("-" * 50)

# Criar figura com subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análise Comparativa: KNN vs SVM - Dataset Iris', fontsize=16, fontweight='bold')

# 1. Gráfico de barras - Comparação de métricas
metrics_comparison = pd.DataFrame({
    'KNN': [knn_cv_results['test_accuracy'].mean(), 
            knn_cv_results['test_precision_macro'].mean(),
            knn_cv_results['test_recall_macro'].mean(),
            knn_cv_results['test_f1_macro'].mean()],
    'SVM': [svm_cv_results['test_accuracy'].mean(),
            svm_cv_results['test_precision_macro'].mean(),
            svm_cv_results['test_recall_macro'].mean(),
            svm_cv_results['test_f1_macro'].mean()]
}, index=['Acurácia', 'Precisão', 'Revocação', 'F1-Score'])

metrics_comparison.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('Comparação de Métricas - Validação Cruzada')
axes[0,0].set_ylabel('Score')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Gráfico de K para KNN
axes[0,1].plot(k_values, knn_scores, marker='o', linewidth=2, markersize=6)
axes[0,1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Melhor K = {best_k}')
axes[0,1].set_title('Acurácia vs Valor de K (KNN)')
axes[0,1].set_xlabel('Valor de K')
axes[0,1].set_ylabel('Acurácia')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Comparação de kernels SVM
kernel_scores = list(svm_results.values())
axes[0,2].bar(kernels, kernel_scores, color=['lightgreen', 'orange', 'purple'])
axes[0,2].set_title('Desempenho por Kernel (SVM)')
axes[0,2].set_ylabel('Acurácia')
axes[0,2].grid(True, alpha=0.3)

# 4. Matriz de confusão KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=axes[1,0])
axes[1,0].set_title('Matriz de Confusão - KNN')
axes[1,0].set_ylabel('Classe Real')
axes[1,0].set_xlabel('Classe Predita')

# 5. Matriz de confusão SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds',
            xticklabels=target_names, yticklabels=target_names, ax=axes[1,1])
axes[1,1].set_title('Matriz de Confusão - SVM')
axes[1,1].set_ylabel('Classe Real')
axes[1,1].set_xlabel('Classe Predita')

# 6. Boxplot da distribuição das métricas
cv_data = pd.DataFrame({
    'KNN_Accuracy': knn_cv_results['test_accuracy'],
    'SVM_Accuracy': svm_cv_results['test_accuracy'],
    'KNN_F1': knn_cv_results['test_f1_macro'],
    'SVM_F1': svm_cv_results['test_f1_macro']
})

cv_data.boxplot(ax=axes[1,2])
axes[1,2].set_title('Distribuição das Métricas - Validação Cruzada')
axes[1,2].set_ylabel('Score')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================================================================================
# 8. ANÁLISE COMPARATIVA FINAL
# ==================================================================================

print("\n8. ANÁLISE COMPARATIVA FINAL")
print("="*80)

# Resumo das métricas de validação cruzada
print("\nRESUMO - VALIDAÇÃO CRUZADA:")
print(f"{'Métrica':<15} {'KNN':<15} {'SVM':<15} {'Diferença':<15}")
print("-" * 60)

metrics_names = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score']
knn_means = [knn_cv_results['test_accuracy'].mean(),
             knn_cv_results['test_precision_macro'].mean(),
             knn_cv_results['test_recall_macro'].mean(),
             knn_cv_results['test_f1_macro'].mean()]

svm_means = [svm_cv_results['test_accuracy'].mean(),
             svm_cv_results['test_precision_macro'].mean(),
             svm_cv_results['test_recall_macro'].mean(),
             svm_cv_results['test_f1_macro'].mean()]

for i, metric in enumerate(metrics_names):
    diff = svm_means[i] - knn_means[i]
    print(f"{metric:<15} {knn_means[i]:<15.4f} {svm_means[i]:<15.4f} {diff:<+15.4f}")

# Determinar o melhor modelo
best_model = "SVM" if np.mean(svm_means) > np.mean(knn_means) else "KNN"
print(f"\nMELHOR MODELO: {best_model}")

# Análise detalhada
print(f"\nANÁLISE DETALHADA:")
print(f"1. KNN (K={best_k}):")
print(f"   - Vantagens: Simples, interpretável, não paramétrico")
print(f"   - Desvantagem: Sensível a outliers e dimensionalidade")
print(f"   - Desempenho: Acurácia média de {knn_cv_results['test_accuracy'].mean():.4f}")

print(f"\n2. SVM (kernel={best_kernel}):")
print(f"   - Vantagens: Eficaz em alta dimensão, versátil com kernels")
print(f"   - Desvantagem: Complexo, requer padronização")
print(f"   - Desempenho: Acurácia média de {svm_cv_results['test_accuracy'].mean():.4f}")

print(f"\nCONCLUSÃO:")
if best_model == "SVM":
    print("SVM apresentou melhor desempenho geral, especialmente devido à sua")
    print("capacidade de encontrar fronteiras de decisão mais complexas.")
else:
    print("KNN apresentou melhor desempenho, demonstrando que a simplicidade")
    print("pode ser eficaz para este dataset bem estruturado.")

print(f"\nAmbos os modelos apresentaram excelente desempenho no dataset Iris,")
print(f"com diferenças mínimas entre as métricas, indicando que o dataset")
print(f"é bem adequado para classificação.")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA")
print("="*80)