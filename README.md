# Exemplos de IA supervisionada e não supervisionada

Projeto da matéria de Inteligência Artificial com dois exemplos em notebooks:

- `notebooks/01_ia_supervisionada_credito.ipynb`
- `notebooks/02_ia_nao_supervisionada_segmentacao.ipynb`

A entrega principal são os notebooks, porque eles juntam explicação, código, validação dos dados, tabelas, métricas e gráficos no mesmo lugar. Isso facilita a apresentação em sala e deixa o raciocínio mais transparente.

## Visão geral

Fato: o projeto tem dois problemas diferentes.

- IA supervisionada: prever se uma decisão de crédito será `aprovado` ou `negado`.
- IA não supervisionada: encontrar grupos de clientes parecidos sem uma coluna pronta de segmento.

Inferência: os exemplos são didáticos, então os resultados servem para explicar conceitos, não para tomar decisões reais de crédito ou marketing.

Opinião técnica: o formato com notebooks é o melhor para essa entrega, porque permite mostrar o fluxo completo de Machine Learning sem alternar entre terminal, script e relatório separado.

## Estrutura principal

```text
.
├── data/
│   ├── dados_clientes_supervisionado.csv
│   └── dados_clientes_nao_supervisionado.csv
├── notebooks/
│   ├── 01_ia_supervisionada_credito.ipynb
│   └── 02_ia_nao_supervisionada_segmentacao.ipynb
├── exemplos/
│   ├── supervisionado_knn_credito.py
│   ├── nao_supervisionado_kmeans_clientes.py
│   └── util_ml.py
├── main.py
├── requirements.txt
├── codex.md
└── README.md
```

## Bibliotecas usadas

- `pandas`, para ler os CSVs e organizar os dados em tabelas.
- `scikit-learn`, para normalização, treino, avaliação e segmentação.
- `plotly`, para gráficos interativos dentro dos notebooks.
- `shap`, para interpretar o modelo supervisionado.
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

## Dados usados

Os notebooks dependem dos CSVs da pasta `data`.

### Base supervisionada

Arquivo:

```text
data/dados_clientes_supervisionado.csv
```

Colunas:

- `renda_mensal`
- `idade`
- `score_credito`
- `divida_atual`
- `decisao_credito`

Fato: a base tem 200 registros, sem valores nulos e sem linhas duplicadas.

Fato: as classes estão balanceadas:

- `aprovado`: 100 registros
- `negado`: 100 registros

### Base não supervisionada

Arquivo:

```text
data/dados_clientes_nao_supervisionado.csv
```

Colunas:

- `cliente`
- `compras_mes`
- `gasto_medio_reais`
- `dias_desde_ultima_compra`

Fato: a base tem 200 registros, sem valores nulos e sem linhas duplicadas.

## Notebook 1, IA supervisionada

Arquivo:

```text
notebooks/01_ia_supervisionada_credito.ipynb
```

O exemplo usa classificação de crédito. Como existe uma coluna alvo, `decisao_credito`, o modelo aprende com respostas conhecidas. Por isso, é IA supervisionada.

Fluxo principal:

- valida a base antes do treino
- mostra boxplots com outliers
- separa treino e teste com estratificação
- treina um `Pipeline` com `MinMaxScaler` e `KNeighborsClassifier`
- exibe acurácia, tabela de métricas e matriz de confusão
- mostra um scatter de renda, score, dívida e decisão
- usa SHAP para explicar a importância das variáveis na aprovação
- simula a previsão de um novo cliente

Resultado atual:

- acurácia aproximada: `0.91`
- classes usadas: `aprovado` e `negado`
- matriz de confusão na ordem `['aprovado', 'negado']`:

```text
[[31, 2],
 [4, 29]]
```

Interpretação prática:

- 31 clientes `aprovado` foram classificados corretamente
- 2 clientes `aprovado` foram classificados como `negado`
- 29 clientes `negado` foram classificados corretamente
- 4 clientes `negado` foram classificados como `aprovado`

Impacto de negócio:

- aprovar errado pode aumentar risco de inadimplência
- negar errado pode gerar perda de receita
- explicar o modelo com SHAP reduz o risco de tratar a decisão como caixa-preta

## Notebook 2, IA não supervisionada

Arquivo:

```text
notebooks/02_ia_nao_supervisionada_segmentacao.ipynb
```

O exemplo usa segmentação de clientes. Como não existe uma coluna dizendo qual é o segmento correto, o algoritmo encontra grupos por semelhança. Por isso, é IA não supervisionada.

Fluxo principal:

- valida a base antes da segmentação
- mostra boxplots com outliers
- normaliza compras, gasto médio e dias desde a última compra
- aplica `KMeans` com 3 grupos
- usa `PCA` para visualizar os clientes em duas dimensões
- calcula `silhouette_score`
- interpreta os centroides como perfis de negócio

Resultado atual:

- coeficiente de silhueta aproximado: `0.49`
- distribuição dos segmentos:

```text
segmento 1: 70 clientes
segmento 2: 49 clientes
segmento 3: 81 clientes
```

Centroides aproximados:

```text
segmento  compras_mes  gasto_medio_reais  dias_desde_ultima_compra
1                4.70             110.57                     54.41
2                1.84              26.63                    126.20
3                8.57             217.52                     22.19
```

Inferência: os grupos indicam perfis coerentes para apresentação:

- risco de abandono, com menos compras, menor gasto e maior tempo sem comprar
- fidelidade média, com comportamento intermediário
- alta fidelidade, com mais compras, maior gasto e compra recente

Impacto de negócio:

- campanhas mais direcionadas
- recuperação de clientes em risco
- melhor alocação do orçamento de marketing
- priorização de clientes com maior potencial de retorno

## Observações importantes

Fato: os dados são fictícios e servem apenas para estudo.

Fato: os notebooks executaram corretamente em uma execução limpa no ambiente virtual.

Inferência: os nomes dos segmentos do K-Means são interpretações humanas dos centroides, não verdades absolutas descobertas pelo algoritmo.

Opinião técnica: em produção, seria necessário usar dados reais, validar viés, evitar data leakage, monitorar desempenho, revisar outliers e criar regras de governança antes de usar qualquer modelo para apoiar decisões.

## Pontos de atenção

- Os arquivos da pasta `data` precisam ser versionados junto com os notebooks, porque os notebooks dependem deles.
- A base supervisionada tem outliers em `divida_atual`, o que é útil para explicação, mas exigiria investigação em produção.
- A silhueta da segmentação é moderada, então o resultado é bom para aula, mas não deve ser apresentado como modelo pronto para negócio real.
- O SHAP aumenta a interpretabilidade, mas adiciona uma dependência mais pesada ao projeto.
