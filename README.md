# Análise de Classificação da Qualidade de Vinhos com Regressão Logística

Este documento detalha o processo de análise de dados e **machine learning** realizado no notebook `Regressão_Logística_Metrics.ipynb`. O objetivo principal é construir e avaliar um modelo de **Regressão Logística** para prever a qualidade de vinhos com base em suas características físico-químicas.

---

## 1. Importação de Bibliotecas

O código inicia importando as bibliotecas essenciais para a análise:

* **numpy e pandas**: Para manipulação de dados e carregamento de datasets.
* **matplotlib.pyplot e seaborn**: Para visualização de dados, criação de gráficos e matriz de confusão.
* **sklearn**: Biblioteca de Machine Learning:

  * `LogisticRegression`: Algoritmo de classificação.
  * `train_test_split`: Para dividir dados em treino e teste.
  * `StandardScaler`: Para normalizar os dados.
  * `confusion_matrix, accuracy_score, metrics`: Para avaliação do modelo.

---

## 2. Carregamento e Pré-processamento dos Dados

### 2.1 Importação do Arquivo .csv

O dataset é carregado usando a função `read_csv` do Pandas:

```python
import pandas as pd

df = pd.read_csv('WineQT.csv')
```

* O arquivo `WineQT.csv` deve estar no mesmo diretório do notebook.
* O DataFrame `df` organiza os dados em linhas e colunas, facilitando manipulação e análise.

### 2.2 Transformação da Variável Alvo

A coluna `quality` possui múltiplos valores (3 a 8). Para classificação binária:

```python
df['quality'] = df.apply(lambda row: 1 if row['quality'] > 5 else 0, axis=1)
```

* **1** → boa qualidade (quality > 5)
* **0** → qualidade ruim (quality ≤ 5)

**Verificação do balanceamento das classes:**

```python
df['quality'].value_counts()
```

* Boa qualidade: 621 vinhos
* Qualidade ruim: 522 vinhos

---

## 3. Preparação para o Treinamento

### 3.1 Seleção de Features e Alvo

```python
X = df.drop('quality', axis=1)
y = df['quality']
```

* `X` contém todas as colunas exceto `quality`.
* `y` contém apenas a coluna `quality`.

### 3.2 Divisão dos Dados

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
```

* 90% dos dados para treinamento e 10% para teste.

---

## 4. Treinamento do Modelo de Regressão Logística

### 4.1 Teste de Diferentes Solvers

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']

for solver in solvers:
    classifier = LogisticRegression(solver=solver)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(f"Acurácia com {solver}: {accuracy_score(y_test, predictions):.4f}")
```

* Melhor solver: **lbfgs**
* Acurácia final: **76,1%**

---

## 5. Avaliação de Performance do Modelo

### 5.1 Matriz de Confusão

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()
```

* Mostra acertos e erros do modelo para cada classe.

### 5.2 Métricas de Classificação

```python
from sklearn import metrics

print(metrics.classification_report(y_test, predictions))
```

* **Acurácia:** 76,1%
* **Precisão:** 79,1%
* **Recall (Sensibilidade):** 76,2%
* **F1-Score:** 77,6%

### 5.3 Curva ROC

* Avalia a capacidade do modelo de distinguir entre vinhos de boa e má qualidade.

---

## 6. Análise de Features

### 6.1 Coeficientes do Modelo

```python
classifier.coef_
```

* Mostra a influência positiva ou negativa de cada variável na previsão da qualidade.

### 6.2 Teste Qui-Quadrado (chi2)

* Avalia a relação entre variáveis de entrada e variável alvo.
* Variáveis como `total sulfur dioxide` e `free sulfur dioxide` apresentam **p-values muito baixos**, indicando forte relação com a qualidade do vinho.

