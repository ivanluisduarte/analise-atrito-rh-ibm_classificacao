'''
Projeto: Análise de atrito de funcionários RH IBM.
Descrição: Projeto de ciência de dados para prever a saída (attrition) de funcionários, 
utilizando o dataset IBM HR Analytics. O objetivo principal é apoiar a área de 
Recursos Humanos (RH) na implementação de estratégias de retenção proativas.
Autor: Ivan Luís Duarte
LinkedIn: https://www.linkedin.com/in/ivanluisduarte/
GitHub: https://github.com/ivanluisduarte
Data de Criação: 2025-10-23
Licença: MIT
'''

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

###############################################################################
from IPython.display import display
from matplotlib.colors import to_rgb, to_rgba
from sklearn.metrics import ConfusionMatrixDisplay

###############################################################################
# ao importar esse arquivo graficos.py, o tema abaixo é aplicado no notebook
contextos = ('paper', 'notebook', 'talk', 'poster')
estilos = ('white', 'dark', 'whitegrid', 'darkgrid', 'ticks')
paletas = ('deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
           'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
           'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c')
sns.set_theme(context=contextos[1], style=estilos[2], palette=paletas[5])


# paleta de cores utilizado nos gráficos/tabelas de comparação do projeto
PALETTE_TEMPERATURA = 'coolwarm'  # 'coolwarm', 'bwr', 'seismic', 'Reds'


###################################################################################
# Função para mostrar e/ou retornar as correlações entre as colunas de um dataframe
###################################################################################
def fnc_correlacao_formatada(
    # DataFrame com os dados (apenas colunas numéricas serão consideradas)
    df: pd.DataFrame,
    flg_mostrar_heatmap: bool = True,  # Se True, mostra o heatmap
    # Se True, mostra também a triangular superior do heatmap
    flg_mostrar_heatmap_triangular_superior: bool = False,
    flg_heatmap_primeiro: bool = True,  # Se True, mostra o heatmap antes do dataframe
    # Se True, mostra o dataframe com as correlações
    flg_mostrar_dataframe: bool = False,
    # Se True, retorna o dataframe com as correlações
    flg_retornar_dataframe: bool = False,
    # Se None, mostra/retorna todas as correlações. Se informado a coluna target,
    coluna_target: str = None,
    # mostra/retorna apenas as correlações com a coluna target no dataframe
    # Paleta de cores para o heatmap ('coolwarm', 'bwr', 'seismic', 'Reds',
    # 'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r')
    palette: str = PALETTE_TEMPERATURA,
    figsize: tuple = (10, 10),  # Tamanho da figura
    precisao: int = 2,  # Número de casas decimais
) -> pd.DataFrame | None:  # Retorna None se flg_retornar_dataframe for False.
    # Retorna o dataframe com as correlações se flg_retornar_dataframe for True
    '''
    Exibe e/ou retorna a matriz de correlação de um DataFrame, opcionalmente incluindo um heatmap.

    Esta função calcula a matriz de correlação de um DataFrame, permitindo a visualização
    através de um heatmap e/ou a exibição/retorno da matriz em formato de DataFrame.
    É possível especificar uma coluna alvo para calcular correlações apenas com ela.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados numéricos.
        flg_mostrar_heatmap (bool, opcional): Se True, exibe o heatmap da matriz de correlação. Padrão: True.
        flg_mostrar_heatmap_triangular_superior (bool, opcional): Se True, exibe também a triangular superior do heatmap.
            Útil quando `flg_mostrar_heatmap` é True. Padrão: False.
        flg_heatmap_primeiro (bool, opcional): Se True, exibe o heatmap antes do DataFrame. Padrão: True.
        flg_mostrar_dataframe (bool, opcional): Se True, exibe a matriz de correlação como um DataFrame. Padrão: False.
        flg_retornar_dataframe (bool, opcional): Se True, retorna a matriz de correlação como um DataFrame. Padrão: False.
        coluna_target (str, opcional): Nome da coluna para calcular correlações apenas com ela. Se None, calcula
            correlações entre todas as colunas. Padrão: None.
        palette (str, opcional): Paleta de cores para o heatmap. Opções: 'coolwarm', 'bwr', 'seismic', 'Reds', 
            'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r'. Padrão: 'coolwarm'.
        figsize (tuple, opcional): Tamanho da figura do heatmap (largura, altura) em polegadas. Padrão: (10, 10).
        precisao (int, opcional): Número de casas decimais para exibir as correlações no DataFrame. Padrão: 2.

    Returns:
        pd.DataFrame | None: Se `flg_retornar_dataframe` for True, retorna a matriz de correlação como um DataFrame.
            Caso contrário, retorna None.

    Raises:
        TypeError: Se `df` não for um DataFrame.

    Examples:
        Para exibir o heatmap e o DataFrame de correlação:
        >>> fnc_correlacao_formatada(df, flg_mostrar_heatmap=True, flg_mostrar_dataframe=True)

        Para retornar apenas o DataFrame de correlação:
        >>> matriz_correlacao = fnc_correlacao_formatada(df, flg_retornar_dataframe=True)

        Para calcular correlações apenas com a coluna 'target':
        >>> fnc_correlacao_formatada(df, coluna_target='target')
    '''

    # Verifica se pelo menos uma das flags de visualização ou retorno está ativa
    if flg_mostrar_heatmap or flg_mostrar_dataframe or flg_retornar_dataframe:
        # Seleciona apenas colunas numéricas e calcula a matriz de correlação
        df_corr = df.select_dtypes(include='number').corr()
        # Se o DataFrame estiver vazio após a seleção de colunas numéricas, retorna None
        if df_corr.empty:
            return None
    else:
        return None  # Se nenhuma opção de visualização ou retorno estiver habilitada, retorna None

    # Função interna para mostrar o heatmap
    def subfnc_mostrar_heatmap():
        if flg_mostrar_heatmap:
            # Cria a figura e o eixo
            fig, ax = plt.subplots(figsize=figsize)

            # Cria o heatmap com seaborn
            sns.heatmap(
                data=df_corr,
                vmin=-1,  # Define o valor mínimo para a escala de cores
                vmax=+1,  # Define o valor máximo para a escala de cores
                mask=None if flg_mostrar_heatmap_triangular_superior else np.triu(
                    df_corr),  # Mascara a parte superior do triângulo se necessário
                annot=True,  # Exibe os valores das correlações nas células
                # Formata os valores com a precisão especificada
                fmt=f'.{precisao}f',
                ax=ax,  # Usa o eixo criado
                cmap=palette  # Usa a paleta de cores especificada
            )

            plt.show()  # Exibe o gráfico

    # Função interna para mostrar o DataFrame

    def subfnc_mostrar_dataframe():
        if flg_mostrar_dataframe:
            # Exibe o DataFrame com estilo condicional usando display
            if coluna_target is None:
                display(df_corr.style.background_gradient(cmap=palette, vmin=-1,
                        # Estilo para todas as colunas
                                                          vmax=1).format(precision=precisao))
            else:
                display(df_corr[[coluna_target]].sort_values(by=[coluna_target]).T.style.background_gradient(
                    # Estilo para coluna target
                    cmap=palette, vmin=-1, vmax=1).format(precision=precisao))

    # Controla a ordem de exibição do heatmap e do DataFrame
    if flg_heatmap_primeiro:
        subfnc_mostrar_heatmap()  # Exibe o heatmap primeiro
        subfnc_mostrar_dataframe()  # Exibe o DataFrame depois
    else:
        subfnc_mostrar_dataframe()  # Exibe o DataFrame primeiro
        subfnc_mostrar_heatmap()  # Exibe o heatmap depois

    # Retorna o DataFrame se a flag estiver ativa
    if flg_retornar_dataframe:
        if coluna_target is None:
            return df_corr  # Retorna o DataFrame completo
        else:
            return df_corr[[coluna_target]]  # Retorna apenas a coluna target
    else:
        return None  # Retorna None se a flag não estiver ativa
###############################################################################


###############################################################################
# Função que gera um gráfico de histograma combinado com outro de boxplot
###############################################################################


def plot_grafico_histograma_boxplot(
        # OBRIGATORIOS
        dataframe: pd.DataFrame  # dataframe contendo a coluna indicada
        , coluna: str  # coluna a ser considerada no dataframe
        # OPCIONAIS
        , hue_destaque: str = None  # se informado divide em um gráfico por categoria
        # AUTOMATICOS=('auto', 'sturges', 'fd', 'doane', 'scott',
        # 'stone', 'rice', 'sqrt') ou QUANTIDADE INTEIRA ou ainda uma LISTA, TUPLA, SERIE ou ARRAY com os limites
        # vazio deixa o intervalo automatico de acordo com os dados na coluna.
        # Somente utilizados para valores laterais.
        , intervalos_x: str | int | list | tuple | pd.Series = 'auto'        # tamanho da figura
        # se o boxplot deve ser apresentado antes do histograma. a proporção dos
        # gráficos é invertida automaticamente se False
        # (boxplot, histograma) permite definir o quanto da figura cada
        # gráfico vai ocupar
        # distancia entre o quado do boxplot e o quadro do histograma
        # String com o titulo da figura
        # String com o titulo daas barras
        # pode ser 'topo', 'centro' ou 'lateral'
        # casas decimais dos valores apresentados nas barras. Não afeta os laterais.
        # grau de transparencia das linhas de grade
        # grau de transparencia do quadro do boxplot
        # grau de transparencia das barras do histograma
        # grau de transparencia da linha de KDE
        , intervalos_y: list | tuple = [], tamanho_figura: tuple = (8, 5), flg_bloxplot_primeiro: bool = False, proporcao_graficos: tuple = (0.15, 0.85), distancia_graficos: float = 0.00, titulo: str = '', y_descricao_barra: str = 'Quantidade', posicao_valores_barras: str = 'topo', casas_decimais: int = 0, transparencia_linha_grade: float = 0.5, transparencia_boxplot: float = 0.7, transparencia_histograma: float = 0.7, transparencia_linha_kde: float = 1.0, cor_titulo: str = 'dimgray'  # cor do título
        , cor_x_descricao: str = 'dimgray'  # cor da descrição do eixo X
        , cor_quadro: str = 'dimgray'  # cor do quadro
        , cor_boxplot: str = None  # cor do quadro do boxplot
        , cor_histograma: str = None  # cor das barras do histograma
        , cor_valores_barras: str = None  # cor dos valores das barras
        , cor_mediana: str = 'C3'  # cor da linha de mediana
        , cor_media: str = 'C4'  # cor da linha de média
        , cor_moda: str = 'C5'  # cor da linha de moda
        , cor_linha_grade: str = 'gray'  # cor das linhas de grade
        , cor_linha_kde: str = None  # cor da linha de KDE

        , estilo_linha_mediana: str = '-'  # tipo/estilo da linha de mediana
        , estilo_linha_media: str = '--'  # tipo/estilo da linha de media
        , estilo_linha_moda: str = '-.'  # tipo/estilo da linha de moda
        , estilo_linha_grade: str = ':'  # tipo/estilo da linha de grade
        # tipo/estilo da linha de KDE ('-', '--', '-.', ':')
        # largura da linha de mediana
        # largura da linha de media
        # largura da linha de moda
        , estilo_linha_kde: str = '-', largura_linha_mediana: float = 2.0, largura_linha_media: float = 1.5, largura_linha_moda: float = 1.0, largura_linha_grade: float = 0.5  # largura das linhas de grade
        , largura_linha_kde: float = 1.5  # largura da linha de KDE

        , tamanho_titulo: int = 16  # tamanho da fonte do titulo
        , tamanho_descricao: int = 14  # tamanho da fonte das descricoes
        , tamanho_valores: int = 10  # tamanho da fonte dos valores
        , tamanho_tick: int = 2  # tamanho do Tick nos eixos

        # se deve ou não apresentar a linha de médiana no histograma
        # se deve ou não apresentar a linha de média nos dois gráficos
        # se deve ou não apresentar a linha de moda nos dois gráficos
        # se as linhas de grade deve aparecer no boxplot
        # se as linhas de grade deve aparecer no histograma
        # se o quadro em volta do gráfico deve ser visível ou não
        # se o KDE padrão deve aparecer.
        # mostra um KDE personalizado que modifica bastante a imagem. Funciona
        , flg_mostrar_mediana: bool = True        # bem com o histograma acumulativo
        , flg_mostrar_media: bool = True, flg_mostrar_moda: bool = False, flg_linha_grade_boxplot: bool = False, flg_linha_grade_histograma: bool = True, flg_mostrar_quadro: bool = True, flg_mostrar_kde: bool = False, flg_kde_personalizado: bool = True, flg_mostrar_legenda: bool = True  # mostra a legenda

        # texto de legenda para o histograma. Se informado ele aparece na legenda
        # texto de legenda para a linha de KDE. Se informado ele aparece na legenda
        # Inteiro com o angulo (-90 a +90) de rotação dos itens do eixo X
        # ('count', 'density', 'frequency', 'probability', 'percent')
        # se o histograma deve ser acumulativo. Quando ativado, tira o KDE padrão.
        # se texto e valores devem aparecer em negrito
        # Opções int ou str = {0='best', 1='upper right', 2='upper left',
        , legenda_histograma: str = None, legenda_kde: str = None, x_rotacao: int = 0, estatistica_histograma: str = 'count', flg_acumulativo: bool = False, flg_negrito: bool = False, posicao_legenda: int | str | tuple = 0
    # 3='lower left', 4='lower right', 5='right', 6='center left', 7='center right', 8='lower center',
    # 9='upper center', 10='center'}. Opções de tupla podem ser (x, y) ou (x, y, w, h).
) -> None:
    '''
        Função que gera um gráfico de histograma combinado com outro de boxplot

        Args:
            # OBRIGATORIOS
            dataframe : pd.DataFrame # dataframe contendo a coluna indicada
            , coluna : str # coluna a ser considerada no dataframe
            # OPCIONAIS
            , hue_destaque : str = None # se informado divide em um gráfico por categoria
            , intervalos_x : str|int|list|tuple|pd.Series = 'auto' # AUTOMATICOS=('auto', 'sturges', 'fd', 'doane', 
                'scott', 'stone', 'rice', 'sqrt') ou QUANTIDADE INTEIRA ou ainda 
                uma LISTA, TUPLA, SERIE ou ARRAY com os limites
            , intervalos_y : list|tuple = [] # vazio deixa o intervalo automatico de acordo com os dados na coluna.
                Somente utilizados para valores laterais.
            , tamanho_figura : tuple = (8, 5) # tamanho da figura
            , flg_bloxplot_primeiro : bool = False # se o boxplot deve ser apresentado antes do histograma. a proporção 
                dos gráficos é invertida automaticamente se False
            , proporcao_graficos : tuple = (0.15, 0.85) # (boxplot, histograma) permite definir o quanto da figura cada 
                gráfico vai ocupar
            , distancia_graficos : float = 0.00 # distancia entre o quado do boxplot e o quadro do histograma

            , titulo : str = '' # String com o titulo da figura
            , y_descricao_barra : str = 'Quantidade' # String com o titulo daas barras

            , posicao_valores_barras : str = 'topo' # pode ser 'topo', 'centro' ou 'lateral'
            , casas_decimais : int = 0 # casas decimais dos valores apresentados nas barras. Não afeta os laterais.

            , transparencia_linha_grade : float = 0.5 # grau de transparencia das linhas de grade
            , transparencia_boxplot : float = 0.7 # grau de transparencia do quadro do boxplot
            , transparencia_histograma : float = 0.7 # grau de transparencia das barras do histograma
            , transparencia_linha_kde : float = 1.0 # grau de transparencia da linha de KDE

            , cor_titulo : str = 'dimgray' # cor do título
            , cor_x_descricao : str = 'dimgray' # cor da descrição do eixo X
            , cor_quadro : str = 'dimgray' # cor do quadro
            , cor_boxplot : str = None # cor do quadro do boxplot
            , cor_histograma : str = None # cor das barras do histograma
            , cor_valores_barras : str = None # cor dos valores das barras
            , cor_mediana : str = 'C3' # cor da linha de mediana
            , cor_media : str = 'C4' # cor da linha de média
            , cor_moda : str = 'C5' # cor da linha de moda
            , cor_linha_grade : str = 'gray' # cor das linhas de grade
            , cor_linha_kde: str = None # cor da linha de KDE

            , estilo_linha_mediana : str = '-' # tipo/estilo da linha de mediana
            , estilo_linha_media : str = '--' # tipo/estilo da linha de media
            , estilo_linha_moda : str = '-.' # tipo/estilo da linha de moda
            , estilo_linha_grade : str = ':' # tipo/estilo da linha de grade
            , estilo_linha_kde : str = '-' # tipo/estilo da linha de KDE ('-', '--', '-.', ':')

            , largura_linha_mediana : float = 2.0 # largura da linha de mediana
            , largura_linha_media : float = 1.5 # largura da linha de media
            , largura_linha_moda : float = 1.0 # largura da linha de moda
            , largura_linha_grade : float = 0.5 # largura das linhas de grade
            , largura_linha_kde : float = 1.5 # largura da linha de KDE

            , tamanho_titulo : int = 16 # tamanho da fonte do titulo
            , tamanho_descricao : int = 14 # tamanho da fonte das descricoes
            , tamanho_valores : int = 10 # tamanho da fonte dos valores
            , tamanho_tick : int = 2 # tamanho do Tick nos eixos

            , flg_mostrar_mediana : bool = True # se deve ou não apresentar a linha de médiana no histograma
            , flg_mostrar_media : bool = True # se deve ou não apresentar a linha de média nos dois gráficos
            , flg_mostrar_moda : bool = False # se deve ou não apresentar a linha de moda nos dois gráficos
            , flg_linha_grade_boxplot : bool = False # se as linhas de grade deve aparecer no boxplot
            , flg_linha_grade_histograma : bool = True # se as linhas de grade deve aparecer no histograma
            , flg_mostrar_quadro : bool = True # se o quadro em volta do gráfico deve ser visível ou não
            , flg_mostrar_kde : bool = False # se o KDE padrão deve aparecer.
            , flg_kde_personalizado : bool = True # mostra um KDE personalizado que modifica bastante a imagem. 
                Funciona com o histograma acumulativo
            , flg_mostrar_legenda : bool = True # mostra a legenda

            , legenda_histograma : str = None # texto de legenda para o histograma. Se informado ele aparece na legenda
            , legenda_kde : str = None # texto de legenda para a linha de KDE. Se informado ele aparece na legenda

            , x_rotacao : int = 0 # Inteiro com o angulo (-90 a +90) de rotação dos itens do eixo X
            , estatistica_histograma : str = 'count' # ('count', 'density', 'frequency', 'probability', 'percent')
            , flg_acumulativo : bool = False # se o histograma deve ser acumulativo. Quando ativado, tira o KDE padrão.
            , flg_negrito : bool = False # se texto e valores devem aparecer em negrito
            , posicao_legenda : int|str|tuple = 0 # Opções int ou str = {0='best', 1='upper right', 2='upper left', 
                3='lower left', 4='lower right', 5='right', 6='center left', 7='center right', 8='lower center', 
                9='upper center', 10='center'}. Opções de tupla podem ser (x, y) ou (x, y, w, h).

        Examples:
            >>> plot_grafico_histograma_boxplot(df, 'coluna')
            Gráfico
    '''

    # <<< Implementação da função >>>

    # figura
    if not flg_bloxplot_primeiro:
        proporcao_graficos = (proporcao_graficos[1], proporcao_graficos[0])

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={
            'height_ratios': proporcao_graficos,
            'hspace': distancia_graficos
        },
        figsize=tamanho_figura
    )

    if flg_bloxplot_primeiro:
        axb = ax1
        axh = ax2
    else:
        axh = ax1
        axb = ax2

    if flg_negrito:
        fontweight = 'bold'
    else:
        fontweight = 'normal'

    if flg_acumulativo:
        flg_mostrar_kde = False

    fig.suptitle(titulo, fontweight=fontweight,
                 color=cor_titulo, fontsize=tamanho_titulo)

    if cor_boxplot:
        cor_boxplot = to_rgba(c=to_rgb(cor_boxplot),
                              alpha=transparencia_boxplot)

    # boxplot
    sns.boxplot(
        data=dataframe,
        x=coluna,
        showmeans=flg_mostrar_media,
        meanline=flg_mostrar_media,
        color=cor_boxplot,
        medianprops={'label': 'Mediana', 'color': cor_mediana,
                     'linewidth': largura_linha_mediana, 'linestyle': estilo_linha_mediana},
        meanprops={'label': 'Média', 'color': cor_media,
                   'linewidth': largura_linha_media, 'linestyle': estilo_linha_media},
        hue=hue_destaque,
        ax=axb,
    )

    # exclui o eixo y da esquerda
    axb.tick_params(axis='y', size=0, left=False, labelleft=False)

    # kde personalizado
    if flg_kde_personalizado:
        axk = axh.twinx()
        sns.kdeplot(data=dataframe, x=coluna, color=cor_linha_kde, ax=axk,
                    label=legenda_kde, hue=hue_destaque, linestyle=estilo_linha_kde,
                    linewidth=largura_linha_kde, alpha=transparencia_linha_kde
                    )
        # exclui o eixo y da esquerda
        axk.tick_params(axis='y', size=0, right=False, labelright=False)
        axk.set_ylabel('')
        flg_mostrar_kde = False
        # estatistica_histograma = 'density'

    # histograma
    hist = sns.histplot(data=dataframe, x=coluna, bins=intervalos_x, ax=axh, hue=hue_destaque,
                        color=cor_histograma, alpha=transparencia_histograma, stat=estatistica_histograma,
                        label=legenda_histograma, kde=flg_mostrar_kde, cumulative=flg_acumulativo
                        )

    # faz rotação no eixo x
    axh.tick_params(axis='x', rotation=x_rotacao,
                    color=cor_x_descricao, size=tamanho_tick)
    # Descrição do eixo X
    axh.set_xlabel(xlabel=coluna, fontweight=fontweight,
                   color=cor_x_descricao, fontsize=tamanho_descricao)

    if type(intervalos_x) not in (str, int):
        axh.set_xticks(intervalos_x)

    # apresentação de valores
    if posicao_valores_barras == 'lateral':
        axh.set_ylabel(ylabel=y_descricao_barra, color=cor_valores_barras,
                       fontweight=fontweight, fontsize=tamanho_descricao)
        axh.tick_params(axis='y', labelcolor=cor_valores_barras,
                        color=cor_valores_barras, size=tamanho_tick)
        if intervalos_y != []:
            axh.set_yticks(intervalos_y)
    else:
        # exclui o eixo y da esquerda
        axh.tick_params(axis='y', size=0, left=False, labelleft=False)
        axh.set_ylabel('')

        if posicao_valores_barras == 'centro':
            posicao_valores_barras = 'center'
        else:
            posicao_valores_barras = 'edge'

        for barra in hist.containers:  # colocando valores nas barras
            hist.bar_label(barra, labels=[f'{b.get_height():,.{casas_decimais}f}'.
                                          replace(',', '_').replace('.', ',').replace('_', '.') for b in barra],
                           color=cor_valores_barras, fontweight=fontweight,
                           label_type=posicao_valores_barras, fontsize=tamanho_valores
                           )

    # linhas de grade
    if flg_linha_grade_boxplot:
        axb.grid(flg_linha_grade_boxplot, linestyle=estilo_linha_grade, color=cor_linha_grade,
                 alpha=transparencia_linha_grade, linewidth=largura_linha_grade)
        axb.set_axisbelow(True)
    if flg_linha_grade_histograma:
        axh.grid(flg_linha_grade_histograma, linestyle=estilo_linha_grade, color=cor_linha_grade,
                 alpha=transparencia_linha_grade, linewidth=largura_linha_grade)
        axh.set_axisbelow(True)

    # linhas de destaque
    if hue_destaque == None:
        if flg_mostrar_mediana:
            axh.axvline(dataframe[coluna].median(), color=cor_mediana, linestyle=estilo_linha_mediana,
                        label='Mediana', linewidth=largura_linha_mediana)
        if flg_mostrar_media:
            axh.axvline(dataframe[coluna].mean(), color=cor_media, linestyle=estilo_linha_media,
                        label='Média', linewidth=largura_linha_media)
        if flg_mostrar_moda:
            axb.axvline(dataframe[coluna].mode()[0], color=cor_moda, linestyle=estilo_linha_moda,
                        label='Moda', linewidth=largura_linha_moda)
            axh.axvline(dataframe[coluna].mode()[0], color=cor_moda, linestyle=estilo_linha_moda,
                        label='Moda', linewidth=largura_linha_moda)

    # posição da legenda
    if flg_mostrar_legenda and not hue_destaque:
        if flg_kde_personalizado and legenda_kde:
            # Combinar as legendas dos dois eixos
            lines_1, labels_1 = axh.get_legend_handles_labels()
            lines_2, labels_2 = axk.get_legend_handles_labels()

            if type(posicao_legenda) == tuple:
                axh.legend(lines_1 + lines_2, labels_1 + labels_2,
                           bbox_to_anchor=posicao_legenda)
            else:
                axh.legend(lines_1 + lines_2, labels_1 +
                           labels_2, loc=posicao_legenda)
        else:
            if type(posicao_legenda) == tuple:
                axh.legend(bbox_to_anchor=posicao_legenda)
            else:
                axh.legend(loc=posicao_legenda)

    lst_ax = list(axb.spines.values()) + list(axh.spines.values())
    if flg_kde_personalizado:
        lst_ax.extend(list(axk.spines.values()))
    for spine in lst_ax:
        if flg_mostrar_quadro:  # quadro de contorno
            spine.set_color(cor_quadro)  # colore o quadro em volta
        else:
            spine.set_visible(False)  # retira o quadro em volta

    plt.show()


########################################################################################################
# função para plotar boxplots de diferentes modelos para diferentes méticas de um dataframe de resultado
########################################################################################################
def plot_comparar_metricas_modelos(
    df_resultados: pd.DataFrame,  # DataFrame contendo os resultados dos modelos
    # Lista/tupla de métricas a serem comparadas
    comparar_metricas: list | tuple = (),
    nomes_metricas: list | tuple = (),  # Nomes para as métricas nos gráficos
    figsize: list | tuple = (10, 8),  # Tamanho da figura (largura, altura)
    colunas_graficos: int = 1,  # Número de colunas de gráficos
    # Se True, plota boxplots horizontalmente
    flg_boxplots_horizontais: bool = False,
    cor_boxplot: str = None,  # Cor dos boxplots
    titulo_grafico: str = 'Comparação de Métricas entre Modelos',  # Título geral do gráfico
    tamanho_titulo: int = 16,  # Tamanho da fonte do título
    tamanho_label: int = 14  # Tamanho da fonte dos labels dos eixos
) -> None:
    '''
    Gera boxplots comparando diferentes modelos em diversas métricas.

    Args:
        df_resultados (pd.DataFrame): DataFrame contendo os resultados dos modelos.
        comparar_metricas (list|tuple, opcional): Lista/tupla de métricas a serem comparadas. 
            Se vazio, compara todas as colunas numéricas.
        nomes_metricas (list|tuple, opcional): Nomes para as métricas nos gráficos. 
            Se não fornecidos, usa os nomes das colunas.
        figsize (list|tuple, opcional): Tamanho da figura (largura, altura) em polegadas.
        colunas_graficos (int, opcional): Número de colunas de gráficos.
        flg_boxplots_horizontais (bool, opcional): Se True, plota boxplots horizontalmente.
        cor_boxplot (str, opcional): Cor dos boxplots. Se None, usa a cor padrão do Seaborn.
        titulo_grafico (str, opcional): Título geral do gráfico.
        tamanho_titulo (int, opcional): Tamanho da fonte do título.
        tamanho_label (int, opcional): Tamanho da fonte dos labels dos eixos.

    Returns:
        None. Exibe os gráficos.

    Raises:
        TypeError: Se `df_resultados` não for um DataFrame.
    '''

    # Verifica se df_resultados é um DataFrame
    if not isinstance(df_resultados, pd.DataFrame):
        raise TypeError('df_resultados deve ser um DataFrame.')

    # Obtém as métricas a serem comparadas
    quantidade_colunas = len(comparar_metricas)

    # Se a lista de métricas estiver vazia, usa todas as colunas numéricas
    if quantidade_colunas == 0:
        comparar_metricas = df_resultados.select_dtypes(
            include='number').columns.to_list()
        quantidade_colunas = len(comparar_metricas)
        # Se não houver colunas numéricas, retorna None
        if quantidade_colunas == 0:
            return None

    # Se os nomes das métricas não forem fornecidos, usa os nomes das colunas
    if len(nomes_metricas) != quantidade_colunas:
        nomes_metricas = comparar_metricas

    # Define o número de colunas de gráficos
    if colunas_graficos <= 0:
        colunas_graficos = 1

    # Calcula o número de linhas de gráficos
    linhas_graficos = ceil(quantidade_colunas / colunas_graficos)

    # Define o compartilhamento dos eixos
    if flg_boxplots_horizontais:
        sharex = False
        sharey = True
    else:
        sharex = True
        sharey = False

    # Cria a figura e os subplots
    fig, axs = plt.subplots(nrows=linhas_graficos, ncols=colunas_graficos,
                            figsize=figsize, sharex=sharex, sharey=sharey)
    # Adiciona o título geral do gráfico
    fig.suptitle(titulo_grafico, fontsize=tamanho_titulo)

    # Itera sobre os subplots, métricas e nomes de métricas
    for ax, metrica, nome_metrica in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        # Define os eixos x e y de acordo com a orientação dos boxplots
        if flg_boxplots_horizontais:
            x = metrica
            y = 'model'
        else:
            x = 'model'
            y = metrica

        # Cria o boxplot
        sns.boxplot(
            data=df_resultados,
            x=x,
            y=y,
            showmeans=True,  # Mostra a média
            ax=ax,
            color=cor_boxplot  # Define a cor do boxplot
        )

        # Configura os labels dos eixos
        if flg_boxplots_horizontais:
            ax.set_xlabel(nome_metrica, fontsize=tamanho_label)
            ax.set_ylabel(None)
        else:
            ax.set_ylabel(nome_metrica, fontsize=tamanho_label)
            ax.set_xlabel(None)

        # Configura o título do subplot
        ax.set_title(nome_metrica, weight='bold', fontsize=tamanho_label)
        # Rotaciona os ticks do eixo x
        ax.tick_params(axis='x', rotation=90, labelsize=tamanho_label)

    # Ajusta o layout para evitar sobreposição, considerando o título geral
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()


###############################################################################
# Gera e plota quatro matrizes de confusão lado a lado (2x2) com diferentes
# opções de normalização: por linha, por coluna, geral e desnormalizada.
###############################################################################
def plot_matrizes_confusao(
    modelo,
    X: pd.DataFrame,
    y: np.ndarray,
    display_labels: list,
    cmap: str = PALETTE_TEMPERATURA,
    figsize: tuple = (11, 11),
    fontsize_titulo_principal: int = 16
):
    """
    Gera e plota quatro matrizes de confusão lado a lado (2x2) com diferentes
    opções de normalização: por linha, por coluna, geral e desnormalizada.

    Parâmetros
    ----------
    modelo
        O objeto estimador treinado.
    X : pd.DataFrame
        Os dados de entrada (features) para predição.
    y : np.ndarray
        Os rótulos verdadeiros (ground truth) para comparação.
    display_labels : list
        Uma lista de strings que representam os nomes das classes
        a serem exibidas na matriz de confusão.
    cmap : str, default=PALETTE_TEMPERATURA
        O mapa de cores (colormap) a ser usado para a visualização.
    figsize : tuple, default=(11, 11)
        O tamanho da figura (largura, altura) em polegadas.
    fontsize_titulo_principal : int, default=16
        O tamanho da fonte para o título principal da figura.

    Retorna
    -------
    None
        Exibe a figura com as matrizes de confusão.
    """
    # Cria a figura e os subplots (eixos) na disposição 2x2
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Define as opções de normalização e os títulos correspondentes
    opcoes_normalizacao = ('true', 'pred', 'all', None)
    titulos_plot = (
        "Normalização por Linha (True)",
        "Normalização por Coluna (Pred)",
        "Normalização Geral (All)",
        "Desnormalizada (None)",
    )
    precisoes = ('.3f', '.3f', '.3f', '.0f')

    # Itera sobre os eixos, normalizações e títulos para plotar cada matriz
    for ax, normalize, titulo, precisao in zip(
        axs.flatten(),  # Transforma a matriz 2x2 de eixos em uma lista plana
        opcoes_normalizacao,
        titulos_plot,
        precisoes,
    ):
        # Gera a matriz de confusão usando estimador
        ConfusionMatrixDisplay.from_estimator(
            estimator=modelo,  # O modelo que será usado para predição
            X=X,  # Dados para predição
            y=y,  # Rótulos verdadeiros
            display_labels=display_labels,  # Nomes das classes
            # Tipo de normalização (True, Pred, All, ou None)
            normalize=normalize,
            cmap=cmap,  # Mapa de cores
            ax=ax,  # Eixo onde o plot será desenhado
            values_format=precisao,  # Formata os valores com 0 casas decimais
        )
        # Define o título do subplot
        ax.set_title(titulo, weight="bold")
        # Remove a grade (grid) para uma visualização mais limpa
        ax.grid(False)

    # Define o título principal da figura
    fig.suptitle("Matrizes de Confusão",
                 fontsize=fontsize_titulo_principal,
                 weight="bold")

    # Ajusta o layout para evitar sobreposição de elementos
    plt.tight_layout()
    # Exibe a figura
    plt.show()
