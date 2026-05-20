from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

from exemplos.nao_supervisionado_kmeans_clientes import executar_segmentacao
from exemplos.supervisionado_knn_credito import treinar_modelo_supervisionado

CAMINHO_RELATORIO = Path('relatorio_ia.html')


def _grafico_credito(resultado_supervisionado: dict[str, Any]) -> str:
    dataframe = resultado_supervisionado['dataframe']
    figura = px.scatter(
        dataframe,
        x='renda_mensal',
        y='score_credito',
        color='decisao_credito',
        size='divida_atual',
        hover_data=['idade', 'divida_atual'],
        color_discrete_map={
            'aprovado': '#2f9e44',
            'negado': '#d9480f',
        },
        title='Clientes por renda, score e decisão de crédito',
    )
    figura.update_layout(
        template='plotly_white',
        legend_title_text='Decisão',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return to_html(figura, full_html=False, include_plotlyjs=True)


def _grafico_matriz_confusao(resultado_supervisionado: dict[str, Any]) -> str:
    classes = resultado_supervisionado['classes']
    matriz_confusao = resultado_supervisionado['matriz_confusao']

    figura = go.Figure(
        data=go.Heatmap(
            z=matriz_confusao,
            x=classes,
            y=classes,
            colorscale=[
                [0.0, '#f8f9fa'],
                [0.5, '#74c0fc'],
                [1.0, '#1864ab'],
            ],
            text=matriz_confusao,
            texttemplate='%{text}',
            hovertemplate='Real: %{y}<br>Previsto: %{x}<br>Quantidade: %{z}<extra></extra>',
        )
    )
    figura.update_layout(
        title='Matriz de confusão do KNN',
        xaxis_title='Previsto',
        yaxis_title='Real',
        template='plotly_white',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return to_html(figura, full_html=False, include_plotlyjs=False)


def _grafico_segmentos(resultado_nao_supervisionado: dict[str, Any]) -> str:
    resultado = resultado_nao_supervisionado['resultado']
    figura = px.scatter(
        resultado,
        x='pca_1',
        y='pca_2',
        color='perfil',
        symbol='segmento',
        size='gasto_medio_reais',
        hover_name='cliente',
        hover_data=['compras_mes', 'gasto_medio_reais', 'dias_desde_ultima_compra'],
        title='Segmentação de clientes visualizada com PCA',
        color_discrete_sequence=['#0ca678', '#f08c00', '#4263eb'],
    )
    figura.update_layout(
        template='plotly_white',
        legend_title_text='Perfil',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return to_html(figura, full_html=False, include_plotlyjs=False)


def _grafico_distribuicao_segmentos(resultado_nao_supervisionado: dict[str, Any]) -> str:
    resultado = resultado_nao_supervisionado['resultado']
    distribuicao = (
        resultado.groupby(['segmento', 'perfil'])
        .size()
        .reset_index(name='clientes')
        .sort_values('segmento')
    )

    figura = px.bar(
        distribuicao,
        x='perfil',
        y='clientes',
        color='perfil',
        text='clientes',
        title='Quantidade de clientes por segmento',
        color_discrete_sequence=['#0ca678', '#f08c00', '#4263eb'],
    )
    figura.update_traces(textposition='outside')
    figura.update_layout(
        template='plotly_white',
        showlegend=False,
        xaxis_title='Perfil encontrado',
        yaxis_title='Clientes',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return to_html(figura, full_html=False, include_plotlyjs=False)


def gerar_relatorio_html(caminho_saida: Path = CAMINHO_RELATORIO) -> Path:
    resultado_supervisionado = treinar_modelo_supervisionado()
    resultado_nao_supervisionado = executar_segmentacao()

    acuracia = resultado_supervisionado['acuracia']
    previsao_novo_cliente = resultado_supervisionado['previsao_novo_cliente']
    probabilidades = resultado_supervisionado['probabilidades_novo_cliente']
    silhueta = resultado_nao_supervisionado['silhueta']
    variancia_pca = resultado_nao_supervisionado['variancia_pca'].sum()

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Relatório de IA supervisionada e não supervisionada</title>
    <style>
        :root {{
            --fundo: #f6f7f9;
            --texto: #20242a;
            --muted: #5c6670;
            --linha: #d9dee5;
            --verde: #0ca678;
            --azul: #4263eb;
            --laranja: #f08c00;
            --vermelho: #d9480f;
            --branco: #ffffff;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: var(--fundo);
            color: var(--texto);
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.5;
        }}

        header {{
            background: #111827;
            color: var(--branco);
            padding: 42px 24px 36px;
            border-bottom: 6px solid var(--verde);
        }}

        main {{
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 48px;
        }}

        h1, h2, h3, p {{
            margin-top: 0;
        }}

        h1 {{
            max-width: 900px;
            margin-bottom: 12px;
            font-size: clamp(2rem, 5vw, 4rem);
            line-height: 1.05;
        }}

        h2 {{
            margin-bottom: 12px;
            font-size: 1.55rem;
        }}

        .subtitulo {{
            max-width: 780px;
            margin-bottom: 0;
            color: #dbe4ff;
            font-size: 1.08rem;
        }}

        .grade-metricas {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 22px;
        }}

        .card {{
            background: var(--branco);
            border: 1px solid var(--linha);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06);
        }}

        .metrica {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--azul);
        }}

        .rotulo {{
            margin-bottom: 6px;
            color: var(--muted);
            font-size: 0.92rem;
        }}

        .secao {{
            margin-top: 24px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
        }}

        .grafico {{
            min-height: 420px;
        }}

        .nota {{
            color: var(--muted);
            font-size: 0.96rem;
        }}

        .tag {{
            display: inline-block;
            margin-right: 8px;
            margin-bottom: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: #e7f5ff;
            color: #1864ab;
            font-size: 0.88rem;
            font-weight: 700;
        }}

        footer {{
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
            padding: 0 0 32px;
            color: var(--muted);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Exemplos de Inteligência Artificial com biblioteca</h1>
        <p class="subtitulo">
            Uma demonstração direta de IA supervisionada e não supervisionada usando scikit-learn,
            pandas e Plotly, com foco em entendimento e apresentação.
        </p>
    </header>

    <main>
        <section class="grade-metricas">
            <article class="card">
                <p class="rotulo">Acurácia do KNN</p>
                <p class="metrica">{acuracia:.0%}</p>
                <p class="nota">Métrica simples para demonstrar acerto no conjunto de teste.</p>
            </article>
            <article class="card">
                <p class="rotulo">Novo cliente</p>
                <p class="metrica">{previsao_novo_cliente}</p>
                <p class="nota">Probabilidades estimadas: {probabilidades}</p>
            </article>
            <article class="card">
                <p class="rotulo">Silhueta do K-Means</p>
                <p class="metrica">{silhueta:.2f}</p>
                <p class="nota">Quanto mais perto de 1, mais separados tendem a estar os grupos.</p>
            </article>
            <article class="card">
                <p class="rotulo">PCA no gráfico</p>
                <p class="metrica">{variancia_pca:.0%}</p>
                <p class="nota">Variação aproximada preservada em duas dimensões para visualização.</p>
            </article>
        </section>

        <section class="card secao">
            <h2>Leitura rápida</h2>
            <p>
                <span class="tag">Fato</span>
                O exemplo supervisionado usa dados com resposta conhecida, então o modelo aprende a prever
                <strong>aprovado</strong> ou <strong>negado</strong>.
            </p>
            <p>
                <span class="tag">Inferência</span>
                No exemplo não supervisionado, os nomes dos segmentos são interpretações dos centroides,
                não rótulos que já existiam na base.
            </p>
            <p>
                <span class="tag">Opinião técnica</span>
                Usar scikit-learn aqui é melhor do que manter tudo manual, porque aproxima a atividade de
                um fluxo real de Machine Learning sem aumentar demais a complexidade.
            </p>
        </section>

        <section class="secao">
            <div class="grid">
                <article class="card grafico">
                    {_grafico_credito(resultado_supervisionado)}
                </article>
                <article class="card grafico">
                    {_grafico_matriz_confusao(resultado_supervisionado)}
                </article>
            </div>
        </section>

        <section class="secao">
            <div class="grid">
                <article class="card grafico">
                    {_grafico_segmentos(resultado_nao_supervisionado)}
                </article>
                <article class="card grafico">
                    {_grafico_distribuicao_segmentos(resultado_nao_supervisionado)}
                </article>
            </div>
        </section>

        <section class="card secao">
            <h2>Impacto prático</h2>
            <p>
                No crédito, o ganho possível está em reduzir análise manual, acelerar triagem e mitigar risco
                de inadimplência. Na segmentação, o valor está em campanhas mais direcionadas, recuperação de
                clientes em risco e melhor uso do orçamento de marketing.
            </p>
            <p class="nota">
                Como os dados são fictícios e pequenos, este material é didático. Em produção, seria necessário
                validar viés, overfitting, data leakage, estabilidade do modelo e regras de governança.
            </p>
        </section>
    </main>

    <footer>
        Relatório gerado automaticamente por <code>gerar_relatorio_html.py</code>.
    </footer>
</body>
</html>
"""

    caminho_saida.write_text(html, encoding='utf-8')
    return caminho_saida.resolve()


if __name__ == '__main__':
    caminho = gerar_relatorio_html()
    print(f'Relatorio HTML gerado em: {caminho}')
