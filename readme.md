# 🌸 Classificação do Dataset Iris - KNN vs SVM

## 📋 Descrição do Projeto

Este projeto implementa uma análise comparativa entre dois algoritmos de classificação (K-Nearest Neighbors e Support Vector Machine) aplicados ao famoso dataset Iris. O objetivo é avaliar o desempenho dos modelos usando validação cruzada e métricas de avaliação abrangentes.

## 🎯 Objetivos

- Implementar classificação com algoritmos KNN e SVM
- Utilizar validação cruzada estratificada (5-fold)
- Calcular métricas de avaliação: acurácia, precisão, revocação e F1-score
- Gerar matrizes de confusão para análise detalhada
- Comparar o desempenho dos dois algoritmos
- Visualizar resultados através de gráficos informativos

## 📊 Dataset

O **Dataset Iris** é um conjunto de dados clássico que contém:
- **150 amostras** de flores íris
- **4 características**: comprimento e largura das sépalas e pétalas
- **3 classes**: Setosa, Versicolor e Virginica
- **50 amostras por classe** (dataset balanceado)

## 🛠️ Tecnologias Utilizadas

- **Python 3.7+**
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Algoritmos de machine learning
- **Matplotlib**: Visualização de dados
- **Seaborn**: Visualização estatística

## 📁 Estrutura do Projeto

```
agenda_06/
├── .venv/                    # Ambiente virtual
├── agenda_06.py              # Código principal
├── requirements.txt          # Dependências
└── README.md                 # Este arquivo
```

## 🚀 Como Executar

### 1. Clonar/Baixar o Projeto

```bash
# Navegar até a pasta do projeto
cd "agenda_06"
```

### 2. Configurar Ambiente Virtual

**Windows (PowerShell):**
```powershell
# Ativar ambiente virtual
.venv\Scripts\activate

# Se der erro de política de execução:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
# Ativar ambiente virtual
source .venv/bin/activate
```

### 3. Instalar Dependências

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# OU instalar manualmente
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 4. Executar o Código

```bash
python agenda_06.py
```

### 5. Desativar Ambiente Virtual

```bash
deactivate
```

## 📈 Resultados Esperados

O programa irá gerar:

1. **Análise Exploratória**: Estatísticas descritivas do dataset
2. **Otimização de Hiperparâmetros**: 
   - Melhor valor de K para KNN
   - Melhor kernel para SVM
3. **Métricas de Validação Cruzada**: Para ambos os modelos
4. **Matrizes de Confusão**: Visualização dos acertos e erros
5. **Gráficos Comparativos**: 6 visualizações diferentes
6. **Análise Comparativa Final**: Conclusões sobre o melhor modelo

## 📊 Métricas Avaliadas

- **Acurácia**: Proporção de predições corretas
- **Precisão**: Proporção de verdadeiros positivos
- **Revocação (Recall)**: Capacidade de encontrar todos os positivos
- **F1-Score**: Média harmônica entre precisão e revocação

## 🔍 Metodologia

### Validação Cruzada Estratificada
- **5-fold cross-validation**
- Mantém a proporção das classes em cada fold
- Reduz o viés na avaliação

### Pré-processamento
- **StandardScaler**: Padronização dos dados
- Importante para KNN (distância) e SVM (otimização)

### Otimização de Hiperparâmetros
- **KNN**: Teste de K de 1 a 20
- **SVM**: Teste de kernels (linear, rbf, poly)

## 📋 Dependências (requirements.txt)

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🎨 Visualizações

O projeto gera 6 gráficos diferentes:

1. **Comparação de Métricas**: Barras comparativas KNN vs SVM
2. **Otimização KNN**: Acurácia vs valor de K
3. **Comparação Kernels**: Desempenho dos kernels SVM
4. **Matriz de Confusão KNN**: Heatmap dos resultados
5. **Matriz de Confusão SVM**: Heatmap dos resultados
6. **Distribuição das Métricas**: Boxplot da validação cruzada

## 🧪 Exemplo de Saída

```
================================================================================
CLASSIFICAÇÃO DO DATASET IRIS - KNN vs SVM
Validação Cruzada e Análise Comparativa
================================================================================

1. INFORMAÇÕES DO DATASET
--------------------------------------------------
Shape dos dados: (150, 4)
Número de classes: 3
Classes: ['setosa' 'versicolor' 'virginica']

...

MELHOR MODELO: SVM
Acurácia média: 0.9733
```

## 🚨 Solução de Problemas

### Erro de Ambiente Virtual (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Erro de Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Erro de Importação
```bash
pip install --user numpy pandas matplotlib seaborn scikit-learn
```

## 📚 Conceitos Abordados

- **Machine Learning Supervisionado**
- **Validação Cruzada**
- **Métricas de Classificação**
- **Comparação de Algoritmos**
- **Visualização de Dados**
- **Pré-processamento de Dados**

## 🏆 Resultados Acadêmicos

Este projeto demonstra:
- ✅ Implementação correta de algoritmos de ML
- ✅ Uso adequado de validação cruzada
- ✅ Cálculo correto de métricas de avaliação
- ✅ Análise comparativa fundamentada
- ✅ Visualizações informativas e profissionais

## 👨‍🎓 Autor

**Disciplina**: Inteligência Artificial  
**Atividade**: Agenda 06 - Classificação com KNN e SVM  
**Valor**: 20% da Média Final

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos.

---

### 💡 Dicas para Execução

1. **Sempre ative o ambiente virtual** antes de executar
2. **Verifique as dependências** se houver erros
3. **O código gera gráficos** - aguarde a execução completa
4. **Resultados podem variar ligeiramente** devido ao random_state

### 🔗 Recursos Adicionais

- [Documentação Scikit-learn](https://scikit-learn.org/stable/)
- [Dataset Iris](https://archive.ics.uci.edu/ml/datasets/iris)
- [Guia de Validação Cruzada](https://scikit-learn.org/stable/modules/cross_validation.html)