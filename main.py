from exemplos.nao_supervisionado_kmeans_clientes import executar_exemplo as executar_nao_supervisionado
from exemplos.supervisionado_knn_credito import executar_exemplo as executar_supervisionado
from gerar_relatorio_html import gerar_relatorio_html


def main() -> None:
    executar_supervisionado()
    executar_nao_supervisionado()
    caminho_relatorio = gerar_relatorio_html()
    print(f'\nRelatorio interativo gerado em: {caminho_relatorio}')


if __name__ == '__main__':
    main()
