from __future__ import annotations

import os

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

try:
    from exemplos.util_ml import imprimir_relatorio_qualidade
except ModuleNotFoundError:
    from util_ml import imprimir_relatorio_qualidade

COLUNAS = ['compras_mes', 'gasto_medio_reais', 'dias_desde_ultima_compra']

DADOS_CLIENTES = [
    {'cliente': 'C01', 'compras_mes': 2, 'gasto_medio_reais': 35, 'dias_desde_ultima_compra': 90},
    {'cliente': 'C02', 'compras_mes': 1, 'gasto_medio_reais': 20, 'dias_desde_ultima_compra': 120},
    {'cliente': 'C03', 'compras_mes': 3, 'gasto_medio_reais': 45, 'dias_desde_ultima_compra': 80},
    {'cliente': 'C04', 'compras_mes': 8, 'gasto_medio_reais': 180, 'dias_desde_ultima_compra': 20},
    {'cliente': 'C05', 'compras_mes': 10, 'gasto_medio_reais': 220, 'dias_desde_ultima_compra': 14},
    {'cliente': 'C06', 'compras_mes': 7, 'gasto_medio_reais': 150, 'dias_desde_ultima_compra': 25},
    {'cliente': 'C07', 'compras_mes': 4, 'gasto_medio_reais': 80, 'dias_desde_ultima_compra': 45},
    {'cliente': 'C08', 'compras_mes': 5, 'gasto_medio_reais': 95, 'dias_desde_ultima_compra': 38},
    {'cliente': 'C09', 'compras_mes': 6, 'gasto_medio_reais': 110, 'dias_desde_ultima_compra': 35},
    {'cliente': 'C10', 'compras_mes': 12, 'gasto_medio_reais': 260, 'dias_desde_ultima_compra': 7},
    {'cliente': 'C11', 'compras_mes': 9, 'gasto_medio_reais': 210, 'dias_desde_ultima_compra': 12},
    {'cliente': 'C12', 'compras_mes': 2, 'gasto_medio_reais': 60, 'dias_desde_ultima_compra': 65},
    {'cliente': 'C13', 'compras_mes': 4, 'gasto_medio_reais': 70, 'dias_desde_ultima_compra': 55},
    {'cliente': 'C14', 'compras_mes': 1, 'gasto_medio_reais': 25, 'dias_desde_ultima_compra': 140},
    {'cliente': 'C15', 'compras_mes': 5, 'gasto_medio_reais': 130, 'dias_desde_ultima_compra': 30},
]


def criar_dataframe_clientes() -> pd.DataFrame:
    return pd.DataFrame(DADOS_CLIENTES)


def interpretar_segmento(centroide: pd.Series) -> str:
    compras_mes = centroide['compras_mes']
    gasto_medio = centroide['gasto_medio_reais']
    dias_sem_comprar = centroide['dias_desde_ultima_compra']

    if compras_mes >= 8 and gasto_medio >= 180 and dias_sem_comprar <= 25:
        return 'alto valor e alta recorrencia'

    if dias_sem_comprar >= 70:
        return 'risco de abandono'

    return 'recorrencia moderada'


def executar_segmentacao() -> dict[str, object]:
    dataframe = criar_dataframe_clientes()
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dataframe[COLUNAS])

    modelo = KMeans(n_clusters=3, random_state=42, n_init=20)
    grupos = modelo.fit_predict(dados_normalizados)

    pca = PCA(n_components=2)
    coordenadas = pca.fit_transform(dados_normalizados)

    resultado = dataframe.copy()
    resultado['segmento'] = grupos + 1
    resultado['pca_1'] = coordenadas[:, 0]
    resultado['pca_2'] = coordenadas[:, 1]

    centroides = pd.DataFrame(
        scaler.inverse_transform(modelo.cluster_centers_),
        columns=COLUNAS,
    )
    centroides['segmento'] = range(1, len(centroides) + 1)
    centroides['perfil'] = centroides.apply(interpretar_segmento, axis=1)

    resultado = resultado.merge(
        centroides[['segmento', 'perfil']],
        on='segmento',
        how='left',
    )

    return {
        'dataframe': dataframe,
        'resultado': resultado,
        'centroides': centroides,
        'silhueta': silhouette_score(dados_normalizados, grupos),
        'variancia_pca': pca.explained_variance_ratio_,
        'modelo': modelo,
        'normalizador': scaler,
    }


def executar_exemplo() -> None:
    dataframe = criar_dataframe_clientes()

    print('\n=== IA nao supervisionada: segmentacao de clientes com scikit-learn ===')
    imprimir_relatorio_qualidade(dataframe[COLUNAS].values.tolist(), COLUNAS)
    print('\nPor que e nao supervisionada: os clientes nao possuem rotulo previo de segmento.')

    resultado = executar_segmentacao()
    centroides = resultado['centroides']

    print('\nModelo usado: MinMaxScaler + KMeans')
    print(f'Coeficiente de silhueta: {resultado["silhueta"]:.2f}')
    print('Segmentos encontrados:')

    for _, linha in centroides.sort_values('segmento').iterrows():
        valores = {coluna: round(linha[coluna], 2) for coluna in COLUNAS}
        quantidade_clientes = int(
            (resultado['resultado']['segmento'] == linha['segmento']).sum()
        )

        print(
            f'- Segmento {int(linha["segmento"])}: '
            f'{quantidade_clientes} clientes, perfil={linha["perfil"]}'
        )
        print(f'  Centroide aproximado: {valores}')

    print('Impacto pratico: permite campanhas mais direcionadas, recuperacao de clientes em risco e melhor uso do budget.')


if __name__ == '__main__':
    executar_exemplo()
