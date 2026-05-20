# Exemplos de IA supervisionada e não supervisionada

Projeto simples para a matéria de Inteligência Artificial, agora com bibliotecas usadas no mercado:

- `scikit-learn` para os modelos de Machine Learning.
- `pandas` para organizar os dados em tabelas.
- `plotly` para gerar um relatório HTML interativo.

## O que tem no projeto

- IA supervisionada: classificação de crédito com KNN.
- IA não supervisionada: segmentação de clientes com K-Means.
- Relatório visual: gráficos interativos em `relatorio_ia.html`.

## Como preparar o ambiente

Crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente no Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como executar

Para rodar tudo e gerar o relatório:

```bash
python main.py
```

Para rodar apenas o exemplo supervisionado:

```bash
python exemplos/supervisionado_knn_credito.py
```

Para rodar apenas o exemplo não supervisionado:

```bash
python exemplos/nao_supervisionado_kmeans_clientes.py
```

Para gerar somente o relatório HTML:

```bash
python gerar_relatorio_html.py
```

Depois, abra o arquivo:

```text
relatorio_ia.html
```

## Exemplo 1, IA supervisionada

Arquivo: `exemplos/supervisionado_knn_credito.py`

O exemplo usa uma base fictícia de clientes com:

- renda mensal
- idade
- score de crédito
- dívida atual
- decisão conhecida, `aprovado` ou `negado`

Como já existe uma resposta correta para cada cliente, o modelo consegue aprender com exemplos anteriores. Por isso é IA supervisionada.

Algoritmo usado:

- `MinMaxScaler`, para normalizar os dados.
- `KNeighborsClassifier`, para classificar o novo cliente.

Impacto prático: esse tipo de modelo pode ajudar a priorizar análise de crédito, reduzir trabalho manual e mitigar risco de inadimplência. Em produção, ele precisaria de governança, auditoria, monitoramento e avaliação de viés.

## Exemplo 2, IA não supervisionada

Arquivo: `exemplos/nao_supervisionado_kmeans_clientes.py`

O exemplo usa uma base fictícia de clientes com:

- compras por mês
- gasto médio
- dias desde a última compra

Não existe uma coluna dizendo qual é o segmento de cada cliente. O modelo encontra grupos parecidos sozinho. Por isso é IA não supervisionada.

Algoritmo usado:

- `MinMaxScaler`, para normalizar os dados.
- `KMeans`, para agrupar clientes.
- `PCA`, para reduzir os dados para duas dimensões e permitir visualização.
- `silhouette_score`, para medir se os grupos ficaram razoavelmente separados.

Impacto prático: esse tipo de solução pode ajudar em campanhas mais direcionadas, recuperação de clientes em risco e melhor uso do orçamento de marketing.

## Relatório visual

Arquivo gerado: `relatorio_ia.html`

O relatório mostra:

- gráfico dos clientes por renda, score e decisão de crédito
- matriz de confusão do KNN
- visualização dos segmentos com PCA
- distribuição de clientes por segmento
- métricas principais em cards
- leitura rápida com fato, inferência e opinião técnica

## Observações importantes

Fato: os dados usados aqui são fictícios e servem apenas para estudo.

Inferência: os nomes dos segmentos do K-Means são interpretações feitas depois que o algoritmo encontra os grupos.

Opinião técnica: usar `scikit-learn` deixa o projeto mais próximo de uma prática real de Machine Learning, sem deixar o código pesado demais para uma atividade acadêmica.
