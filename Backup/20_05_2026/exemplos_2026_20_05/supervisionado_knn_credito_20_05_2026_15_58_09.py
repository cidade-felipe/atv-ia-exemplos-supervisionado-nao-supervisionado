from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

try:
    from exemplos.util_ml import imprimir_relatorio_qualidade
except ModuleNotFoundError:
    from util_ml import imprimir_relatorio_qualidade

COLUNAS = ['renda_mensal', 'idade', 'score_credito', 'divida_atual']
COLUNA_ALVO = 'decisao_credito'

DADOS_CLIENTES = [
    {'renda_mensal': 3200, 'idade': 22, 'score_credito': 580, 'divida_atual': 2500, 'decisao_credito': 'negado'},
    {'renda_mensal': 7200, 'idade': 35, 'score_credito': 760, 'divida_atual': 1200, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 4500, 'idade': 29, 'score_credito': 640, 'divida_atual': 3200, 'decisao_credito': 'negado'},
    {'renda_mensal': 8900, 'idade': 41, 'score_credito': 810, 'divida_atual': 900, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 5100, 'idade': 31, 'score_credito': 690, 'divida_atual': 1800, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 2700, 'idade': 24, 'score_credito': 520, 'divida_atual': 4200, 'decisao_credito': 'negado'},
    {'renda_mensal': 6300, 'idade': 38, 'score_credito': 710, 'divida_atual': 2100, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 3900, 'idade': 27, 'score_credito': 610, 'divida_atual': 3500, 'decisao_credito': 'negado'},
    {'renda_mensal': 7600, 'idade': 45, 'score_credito': 780, 'divida_atual': 1500, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 3400, 'idade': 36, 'score_credito': 590, 'divida_atual': 2800, 'decisao_credito': 'negado'},
    {'renda_mensal': 5800, 'idade': 33, 'score_credito': 700, 'divida_atual': 1600, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 4200, 'idade': 25, 'score_credito': 600, 'divida_atual': 3900, 'decisao_credito': 'negado'},
    {'renda_mensal': 6800, 'idade': 30, 'score_credito': 730, 'divida_atual': 2200, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 3100, 'idade': 28, 'score_credito': 550, 'divida_atual': 3600, 'decisao_credito': 'negado'},
    {'renda_mensal': 9300, 'idade': 49, 'score_credito': 830, 'divida_atual': 1100, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 3700, 'idade': 39, 'score_credito': 570, 'divida_atual': 4100, 'decisao_credito': 'negado'},
    {'renda_mensal': 5600, 'idade': 26, 'score_credito': 680, 'divida_atual': 2400, 'decisao_credito': 'aprovado'},
    {'renda_mensal': 2500, 'idade': 21, 'score_credito': 500, 'divida_atual': 3800, 'decisao_credito': 'negado'},
]


def criar_dataframe_clientes() -> pd.DataFrame:
    return pd.DataFrame(DADOS_CLIENTES)


def treinar_modelo_supervisionado() -> dict[str, object]:
    dataframe = criar_dataframe_clientes()
    dados = dataframe[COLUNAS]
    alvo = dataframe[COLUNA_ALVO]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        dados,
        alvo,
        test_size=0.33,
        random_state=42,
        stratify=alvo,
    )

    modelo = Pipeline(
        steps=[
            ('normalizacao', MinMaxScaler()),
            ('classificador', KNeighborsClassifier(n_neighbors=3)),
        ]
    )
    modelo.fit(x_treino, y_treino)

    previsoes = modelo.predict(x_teste)
    probabilidades = modelo.predict_proba(x_teste)
    classes = list(modelo.named_steps['classificador'].classes_)

    resultado_teste = x_teste.copy()
    resultado_teste['real'] = y_teste.to_list()
    resultado_teste['previsto'] = previsoes

    for indice_classe, classe in enumerate(classes):
        resultado_teste[f'probabilidade_{classe}'] = probabilidades[:, indice_classe]

    novo_cliente = pd.DataFrame(
        [
            {
                'renda_mensal': 6200,
                'idade': 34,
                'score_credito': 705,
                'divida_atual': 1900,
            }
        ]
    )
    previsao_novo_cliente = modelo.predict(novo_cliente)[0]
    probabilidades_novo_cliente = dict(
        zip(classes, modelo.predict_proba(novo_cliente)[0].round(3))
    )

    return {
        'dataframe': dataframe,
        'modelo': modelo,
        'acuracia': accuracy_score(y_teste, previsoes),
        'matriz_confusao': confusion_matrix(y_teste, previsoes, labels=classes),
        'classes': classes,
        'relatorio_classificacao': classification_report(
            y_teste,
            previsoes,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
        'resultado_teste': resultado_teste,
        'novo_cliente': novo_cliente.iloc[0].to_dict(),
        'previsao_novo_cliente': previsao_novo_cliente,
        'probabilidades_novo_cliente': probabilidades_novo_cliente,
    }


def executar_exemplo() -> None:
    dataframe = criar_dataframe_clientes()

    print('\n=== IA supervisionada: classificacao de credito com scikit-learn ===')
    imprimir_relatorio_qualidade(dataframe[COLUNAS].values.tolist(), COLUNAS)
    print('\nPor que e supervisionada: cada cliente do treino ja possui o rotulo aprovado ou negado.')

    resultado = treinar_modelo_supervisionado()
    resultado_teste = resultado['resultado_teste']

    print(f'\nModelo usado: Pipeline(MinMaxScaler + KNeighborsClassifier)')
    print(f'Acuracia no teste: {resultado["acuracia"]:.0%}')
    print('Comparacao real x previsto:')

    for indice, linha in resultado_teste.reset_index(drop=True).iterrows():
        print(
            f'- Cliente de teste {indice + 1}: '
            f'real={linha["real"]}, previsto={linha["previsto"]}'
        )

    print('\nNovo cliente analisado:')
    print(resultado['novo_cliente'])
    print(f'Previsao do modelo: {resultado["previsao_novo_cliente"]}')
    print(f'Probabilidades: {resultado["probabilidades_novo_cliente"]}')
    print('Impacto pratico: ajuda a priorizar analises, reduzir trabalho manual e mitigar risco de inadimplencia.')


if __name__ == '__main__':
    executar_exemplo()
