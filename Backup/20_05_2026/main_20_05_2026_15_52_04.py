from exemplos.nao_supervisionado_kmeans_clientes import executar_exemplo as executar_nao_supervisionado
from exemplos.supervisionado_knn_credito import executar_exemplo as executar_supervisionado


def main() -> None:
    executar_supervisionado()
    executar_nao_supervisionado()


if __name__ == '__main__':
    main()
