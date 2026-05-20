from __future__ import annotations

import math
from statistics import median

Vetor = list[float]
Matriz = list[Vetor]


def validar_matriz_numerica(
    dados: Matriz,
    colunas: list[str],
    *,
    permitir_negativos: bool = False,
) -> None:
    if not dados:
        raise ValueError('A base de dados nao pode estar vazia.')

    for indice_linha, linha in enumerate(dados, start=1):
        if len(linha) != len(colunas):
            raise ValueError(
                f'Linha {indice_linha} possui {len(linha)} valores, '
                f'mas eram esperados {len(colunas)}.'
            )

        for indice_coluna, valor in enumerate(linha):
            coluna = colunas[indice_coluna]

            if valor is None:
                raise ValueError(f'Valor nulo encontrado em {coluna}, linha {indice_linha}.')

            if isinstance(valor, bool) or not isinstance(valor, (int, float)):
                raise TypeError(
                    f'Valor invalido em {coluna}, linha {indice_linha}: '
                    'esperado numero inteiro ou decimal.'
                )

            if not math.isfinite(valor):
                raise ValueError(f'Valor infinito ou NaN em {coluna}, linha {indice_linha}.')

            if not permitir_negativos and valor < 0:
                raise ValueError(f'Valor negativo em {coluna}, linha {indice_linha}.')


def calcular_minimos_maximos(dados: Matriz) -> list[tuple[float, float]]:
    return [(min(coluna), max(coluna)) for coluna in zip(*dados)]


def normalizar_linha(linha: Vetor, minimos_maximos: list[tuple[float, float]]) -> Vetor:
    linha_normalizada: Vetor = []

    for valor, (valor_minimo, valor_maximo) in zip(linha, minimos_maximos):
        intervalo = valor_maximo - valor_minimo
        linha_normalizada.append(0.0 if intervalo == 0 else (valor - valor_minimo) / intervalo)

    return linha_normalizada


def normalizar_matriz(
    dados: Matriz,
    minimos_maximos: list[tuple[float, float]] | None = None,
) -> tuple[Matriz, list[tuple[float, float]]]:
    parametros = minimos_maximos or calcular_minimos_maximos(dados)
    return [normalizar_linha(linha, parametros) for linha in dados], parametros


def distancia_euclidiana(primeiro_vetor: Vetor, segundo_vetor: Vetor) -> float:
    return math.sqrt(
        sum((primeiro_valor - segundo_valor) ** 2 for primeiro_valor, segundo_valor in zip(primeiro_vetor, segundo_vetor))
    )


def contar_outliers_iqr(dados: Matriz, colunas: list[str]) -> dict[str, int]:
    outliers: dict[str, int] = {}

    for indice_coluna, coluna in enumerate(colunas):
        valores = sorted(linha[indice_coluna] for linha in dados)

        if len(valores) < 4:
            outliers[coluna] = 0
            continue

        meio = len(valores) // 2
        metade_inferior = valores[:meio]
        metade_superior = valores[meio:] if len(valores) % 2 == 0 else valores[meio + 1 :]

        q1 = median(metade_inferior)
        q3 = median(metade_superior)
        intervalo_iqr = q3 - q1
        limite_inferior = q1 - 1.5 * intervalo_iqr
        limite_superior = q3 + 1.5 * intervalo_iqr

        outliers[coluna] = sum(
            1 for valor in valores if valor < limite_inferior or valor > limite_superior
        )

    return outliers


def imprimir_relatorio_qualidade(dados: Matriz, colunas: list[str]) -> None:
    validar_matriz_numerica(dados, colunas)
    outliers = contar_outliers_iqr(dados, colunas)

    print('Validacao dos dados:')
    print(f'- Registros analisados: {len(dados)}')
    print('- Tipos: todas as colunas numericas foram validadas')
    print('- Valores nulos: nenhum valor nulo encontrado')
    print('- Inconsistencias: nenhuma inconsistencia estrutural encontrada')
    print(f'- Outliers por IQR: {outliers}')
