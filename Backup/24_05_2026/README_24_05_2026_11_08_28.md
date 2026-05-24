# Exemplos de IA supervisionada e não supervisionada

Projeto para a matéria de Inteligência Artificial com dois notebooks:

- `notebooks/01_ia_supervisionada_credito.ipynb`
- `notebooks/02_ia_nao_supervisionada_segmentacao.ipynb`

A entrega principal agora são os notebooks, porque fica mais fácil explicar em sala: o código, as tabelas, as métricas e os gráficos aparecem no mesmo lugar.

## Bibliotecas usadas

- `scikit-learn`, para os modelos de Machine Learning.
- `pandas`, para organizar os dados em tabelas.
- `plotly`, para gráficos interativos dentro dos notebooks.
- `ipykernel` e `nbformat`, para facilitar a execução dos notebooks no ambiente virtual.

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

## Como usar

Abra um notebook por vez:

```text
notebooks/01_ia_supervisionada_credito.ipynb
notebooks/02_ia_nao_supervisionada_segmentacao.ipynb
```

Depois execute as células de cima para baixo.

Se quiser rodar os exemplos pelo terminal, use:

```bash
python main.py
```

## Notebook 1, IA supervisionada

Arquivo: `notebooks/01_ia_supervisionada_credito.ipynb`

O exemplo usa uma base fictícia de crédito com:

- renda mensal
- idade
- score de crédito
- dívida atual
- decisão de crédito, `aprovado` ou `negado`

Fato: existe uma resposta correta na base, a coluna `decisao_credito`.

Por isso, o modelo é supervisionado.

O notebook usa:

- `MinMaxScaler`, para normalizar os dados.
- `KNeighborsClassifier`, para classificar o cliente.
- `accuracy_score`, para medir acurácia.
- `confusion_matrix`, para montar a matriz de confusão.
- `plotly`, para mostrar os gráficos no próprio notebook.

Impacto prático: esse tipo de solução pode ajudar a reduzir análise manual de crédito, acelerar triagem e mitigar risco de inadimplência. Em produção, precisaria de auditoria, análise de viés e regras de governança.

## Notebook 2, IA não supervisionada

Arquivo: `notebooks/02_ia_nao_supervisionada_segmentacao.ipynb`

O exemplo usa uma base fictícia de clientes com:

- compras por mês
- gasto médio
- dias desde a última compra

Fato: não existe uma coluna dizendo o segmento do cliente.

Por isso, o modelo é não supervisionado.

O notebook usa:

- `MinMaxScaler`, para normalizar os dados.
- `KMeans`, para agrupar clientes parecidos.
- `PCA`, para visualizar os grupos em duas dimensões.
- `silhouette_score`, para avaliar se os grupos ficaram razoavelmente separados.
- `plotly`, para mostrar os gráficos no próprio notebook.

Impacto prático: esse tipo de solução ajuda em campanhas mais direcionadas, recuperação de clientes em risco e melhor uso do orçamento de marketing.

## Observações importantes

Fato: os dados são fictícios e servem apenas para estudo.

Inferência: os nomes dos segmentos do K-Means são interpretações feitas depois que o algoritmo encontra os grupos.

Opinião técnica: notebooks são a melhor escolha para apresentar esse trabalho em pouco tempo, porque evitam alternar entre terminal, script e HTML separado.
