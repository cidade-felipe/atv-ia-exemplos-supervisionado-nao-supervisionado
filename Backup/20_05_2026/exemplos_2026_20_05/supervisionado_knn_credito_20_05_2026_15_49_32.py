from __future__ import annotations

from collections import Counter, defaultdict

from exemplos.util_ml import (
    Matriz,
    Vetor,
    distancia_euclidiana,
    imprimir_relatorio_qualidade,
    normalizar_linha,
    normalizar_matriz,
)

COLUNAS = ['renda_mensal', 'idade', 'score_credito', 'divida_atual']

DADOS_CLIENTES: list[tuple[Vetor, str]] = [
    ([3200, 22, 580, 2500], 'negado'),
    ([7200, 35, 760, 1200], 'aprovado'),
    ([4500, 29, 640, 3200], 'negado'),
    ([8900, 41, 810, 900], 'aprovado'),
    ([5100, 31, 690, 1800], 'aprovado'),
    ([2700, 24, 520, 4200], 'negado'),
    ([6300, 38, 710, 2100], 'aprovado'),
    ([3900, 27, 610, 3500], 'negado'),
    ([7600, 45, 780, 1500], 'aprovado'),
    ([3400, 36, 590, 2800], 'negado'),
    ([5800, 33, 700, 1600], 'aprovado'),
    ([4200, 25, 600, 3900], 'negado'),
    ([6800, 30, 730, 2200], 'aprovado'),
    ([3100, 28, 550, 3600], 'negado'),
    ([9300, 49, 830, 1100], 'aprovado'),
    ([3700, 39, 570, 4100], 'negado'),
    ([5600, 26, 680, 2400], 'aprovado'),
    ([2500, 21, 500, 3800], 'negado'),
]


def separar_treino_teste(
    dados: list[tuple[Vetor, str]],
    quantidade_teste: int = 6,
) -> tuple[list[tuple[Vetor, str]], list[tuple[Vetor, str]]]:
    if quantidade_teste <= 0 or quantidade_teste >= len(dados):
        raise ValueError('A quantidade de teste precisa deixar exemplos para treino e teste.')

    return dados[:-quantidade_teste], dados[-quantidade_teste:]


def validar_rotulos(dados: list[tuple[Vetor, str]]) -> None:
    rotulos_validos = {'aprovado', 'negado'}

    for indice, (_, rotulo) in enumerate(dados, start=1):
        if rotulo not in rotulos_validos:
            raise ValueError(f'Rotulo invalido na linha {indice}: {rotulo}')


def prever_knn(
    dados_treino: Matriz,
    rotulos_treino: list[str],
    cliente: Vetor,
    *,
    k: int = 3,
) -> str:
    distancias = [
        (distancia_euclidiana(cliente_treino, cliente), rotulo)
        for cliente_treino, rotulo in zip(dados_treino, rotulos_treino)
    ]
    vizinhos = sorted(distancias, key=lambda item: item[0])[:k]

    votos = Counter(rotulo for _, rotulo in vizinhos)
    distancia_media_por_rotulo: dict[str, float] = defaultdict(float)

    for rotulo in votos:
        distancias_do_rotulo = [distancia for distancia, voto in vizinhos if voto == rotulo]
        distancia_media_por_rotulo[rotulo] = sum(distancias_do_rotulo) / len(distancias_do_rotulo)

    return sorted(votos, key=lambda rotulo: (-votos[rotulo], distancia_media_por_rotulo[rotulo]))[0]


def avaliar_modelo(
    dados_treino_normalizados: Matriz,
    rotulos_treino: list[str],
    dados_teste_normalizados: Matriz,
    rotulos_teste: list[str],
) -> tuple[float, list[tuple[str, str]]]:
    previsoes: list[tuple[str, str]] = []

    for cliente, rotulo_real in zip(dados_teste_normalizados, rotulos_teste):
        rotulo_previsto = prever_knn(dados_treino_normalizados, rotulos_treino, cliente)
        previsoes.append((rotulo_real, rotulo_previsto))

    acertos = sum(1 for rotulo_real, rotulo_previsto in previsoes if rotulo_real == rotulo_previsto)
    return acertos / len(previsoes), previsoes


def executar_exemplo() -> None:
    dados = [cliente for cliente, _ in DADOS_CLIENTES]
    validar_rotulos(DADOS_CLIENTES)

    print('\n=== IA supervisionada: classificacao de credito com KNN ===')
    imprimir_relatorio_qualidade(dados, COLUNAS)
    print('\nPor que e supervisionada: cada cliente do treino ja possui o rotulo aprovado ou negado.')

    treino, teste = separar_treino_teste(DADOS_CLIENTES)
    dados_treino = [cliente for cliente, _ in treino]
    rotulos_treino = [rotulo for _, rotulo in treino]
    dados_teste = [cliente for cliente, _ in teste]
    rotulos_teste = [rotulo for _, rotulo in teste]

    dados_treino_normalizados, parametros_normalizacao = normalizar_matriz(dados_treino)
    dados_teste_normalizados, _ = normalizar_matriz(dados_teste, parametros_normalizacao)

    acuracia, previsoes = avaliar_modelo(
        dados_treino_normalizados,
        rotulos_treino,
        dados_teste_normalizados,
        rotulos_teste,
    )

    print(f'\nAcuracia no teste: {acuracia:.0%}')
    print('Comparacao real x previsto:')
    for indice, (rotulo_real, rotulo_previsto) in enumerate(previsoes, start=1):
        print(f'- Cliente de teste {indice}: real={rotulo_real}, previsto={rotulo_previsto}')

    novo_cliente = [6200, 34, 705, 1900]
    novo_cliente_normalizado = normalizar_linha(novo_cliente, parametros_normalizacao)
    previsao = prever_knn(dados_treino_normalizados, rotulos_treino, novo_cliente_normalizado)

    print('\nNovo cliente analisado:')
    print(dict(zip(COLUNAS, novo_cliente)))
    print(f'Previsao do modelo: {previsao}')
    print('Impacto pratico: ajuda a priorizar analises, reduzir trabalho manual e mitigar risco de inadimplencia.')


if __name__ == '__main__':
    executar_exemplo()
