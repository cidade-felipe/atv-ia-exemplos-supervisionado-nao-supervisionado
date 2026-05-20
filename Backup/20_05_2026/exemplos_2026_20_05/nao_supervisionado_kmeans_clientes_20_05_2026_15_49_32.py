from __future__ import annotations

from exemplos.util_ml import (
    Matriz,
    Vetor,
    distancia_euclidiana,
    imprimir_relatorio_qualidade,
    normalizar_matriz,
)

COLUNAS = ['compras_mes', 'gasto_medio_reais', 'dias_desde_ultima_compra']

DADOS_CLIENTES: Matriz = [
    [2, 35, 90],
    [1, 20, 120],
    [3, 45, 80],
    [8, 180, 20],
    [10, 220, 14],
    [7, 150, 25],
    [4, 80, 45],
    [5, 95, 38],
    [6, 110, 35],
    [12, 260, 7],
    [9, 210, 12],
    [2, 60, 65],
    [4, 70, 55],
    [1, 25, 140],
    [5, 130, 30],
]


def calcular_centroide(pontos: Matriz, dimensoes: int) -> Vetor:
    if not pontos:
        return [0.0] * dimensoes

    return [
        sum(ponto[indice] for ponto in pontos) / len(pontos)
        for indice in range(dimensoes)
    ]


def executar_kmeans(
    dados: Matriz,
    centroides_iniciais: Matriz,
    *,
    max_iteracoes: int = 20,
) -> tuple[list[int], Matriz]:
    centroides = [centroide[:] for centroide in centroides_iniciais]
    atribuicoes = [-1] * len(dados)

    for _ in range(max_iteracoes):
        novas_atribuicoes = []

        for cliente in dados:
            distancias = [distancia_euclidiana(cliente, centroide) for centroide in centroides]
            novas_atribuicoes.append(distancias.index(min(distancias)))

        if novas_atribuicoes == atribuicoes:
            break

        atribuicoes = novas_atribuicoes
        novos_centroides: Matriz = []

        for indice_grupo in range(len(centroides)):
            pontos_do_grupo = [
                cliente for cliente, grupo in zip(dados, atribuicoes) if grupo == indice_grupo
            ]
            novos_centroides.append(calcular_centroide(pontos_do_grupo, len(dados[0])))

        centroides = novos_centroides

    return atribuicoes, centroides


def desfazer_normalizacao(
    centroide: Vetor,
    parametros_normalizacao: list[tuple[float, float]],
) -> Vetor:
    valores_originais: Vetor = []

    for valor, (valor_minimo, valor_maximo) in zip(centroide, parametros_normalizacao):
        valores_originais.append(valor_minimo + valor * (valor_maximo - valor_minimo))

    return valores_originais


def interpretar_segmento(centroide_original: Vetor) -> str:
    compras_mes, gasto_medio, dias_sem_comprar = centroide_original

    if compras_mes >= 8 and gasto_medio >= 180 and dias_sem_comprar <= 25:
        return 'alto valor e alta recorrencia'

    if dias_sem_comprar >= 70:
        return 'risco de abandono'

    return 'recorrencia moderada'


def executar_exemplo() -> None:
    print('\n=== IA nao supervisionada: segmentacao de clientes com K-Means ===')
    imprimir_relatorio_qualidade(DADOS_CLIENTES, COLUNAS)
    print('\nPor que e nao supervisionada: os clientes nao possuem rotulo previo de segmento.')

    dados_normalizados, parametros_normalizacao = normalizar_matriz(DADOS_CLIENTES)
    centroides_iniciais = [
        dados_normalizados[1],
        dados_normalizados[7],
        dados_normalizados[9],
    ]

    atribuicoes, centroides_normalizados = executar_kmeans(dados_normalizados, centroides_iniciais)

    print('\nSegmentos encontrados:')
    for indice_grupo, centroide_normalizado in enumerate(centroides_normalizados, start=1):
        centroide_original = desfazer_normalizacao(centroide_normalizado, parametros_normalizacao)
        quantidade_clientes = sum(1 for grupo in atribuicoes if grupo == indice_grupo - 1)
        interpretacao = interpretar_segmento(centroide_original)
        valores = {
            coluna: round(valor, 2)
            for coluna, valor in zip(COLUNAS, centroide_original)
        }

        print(f'- Segmento {indice_grupo}: {quantidade_clientes} clientes, perfil={interpretacao}')
        print(f'  Centroide aproximado: {valores}')

    print('Impacto pratico: permite campanhas mais direcionadas, recuperacao de clientes em risco e melhor uso do budget.')


if __name__ == '__main__':
    executar_exemplo()
