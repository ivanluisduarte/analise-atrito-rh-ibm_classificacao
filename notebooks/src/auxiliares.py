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

import pandas as pd
from IPython.display import display
from kagglehub import (  # para carregar dados do Kaggle
    KaggleDatasetAdapter,
    load_dataset,
)
from scipy.stats import kstest, norm, shapiro  # para testes de normalidade

from .graficos import PALETTE_TEMPERATURA

###########################################################################################################################################################
# Função que retorna um dataframe com o describe() de variáveis numéricas enriquecido com colunas de tipo, ausência, variabilidade, outliers e normalidade.
###########################################################################################################################################################

# Altere o dicionário 'dct_colunas_coloridas' dentro da função abaixo [fnc_describe_numericas_personalizado] para manipular o resultado.
# Para eliminar colunas, comente o indesejado no dicionário do inicio da função.


def fnc_describe_numericas_personalizado(
        df: pd.DataFrame,  # dataframe para o describe() de variáveis numéricas
        # True para mostrar o dataframe formatado, False para não mostrar
        flg_mostrar_dataframe: bool = True,
        # True para retornar o dataframe para uso posterior, False para não retornar
        flg_retornar_dataframe: bool = False,
        # Se 'automatico', usa 'Shapiro-Wilk' para até 5 mil observações e 'Kolmogorov-Smirnov' para quantidades superiores
        teste_normalidade: str = 'automatico',
        # Nivel de significancia para o teste de normalidade Shapiro-Wilk ou Kolmogorov-Smirnov
        nivel_significancia: float = 0.05,
        # 'coolwarm', 'bwr', 'seismic', 'Reds', 'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'RdYlGn_r'
        palette: str = PALETTE_TEMPERATURA,
) -> None | pd.DataFrame:  # retorna None ou o dataframe de acordo com o parâmetro flg_retornar_dataframe
    '''
        Função que retorna um dataframe com o describe() de variáveis numéricas
        enriquecido com colunas de tipo, ausência, variabilidade, outliers e normalidade.

        Altere o dicionário 'dct_colunas_coloridas' dentro da função para manipular o resultado.

        Args:
            df: pd.DataFrame -> dataframe para o describe() de variáveis numéricas
            flg_mostrar_dataframe: bool= True -> True para mostrar o dataframe formatado, False para não mostrar
            flg_retornar_dataframe: bool= False -> True para retornar o dataframe para uso posterior, False para não retornar
            teste_normalidade: str -> default='automatico', usa 'Shapiro-Wilk' para até 5 mil observações e 'Kolmogorov-Smirnov' (KS) para quantidades superiores
            nivel_significancia: float -> default=0.05, Nivel de significancia para o teste de normalidade Shapiro-Wilk ou KS
            palette: str -> default='coolwarm' - outras opções: 'coolwarm', 'bwr', 'seismic', 'Reds', 'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'RdYlGn_r'

        Returns:
            None|pd.DataFrame -> retorna None ou o dataframe de acordo com o parâmetro flg_retornar_dataframe

        Examples:
            >>> fnc_describe_numericas_personalizado(df)
            dataframe com o describe() de variáveis numéricas
    '''

    ############################################################################
    ### este dicionário determina a presença e ordem das colunas,           ####
    ### também determina se deve ou não participar da coloração por 'calor' ####
    ############################################################################
    dct_colunas_coloridas = {
        'tipo': False,  # tipo de dado da coluna
        'percentual inteiros': False,  # percentual de inteiros na coluna
        'nulo': False,  # Quantidades de nulos/NaN
        'count': False,  # Quantidade de registros
        'nulo/count': False,  # Quantidades de nulos / Quantidade de registros
        'soma': False,  # Soma total dos valores da coluna
        'unicos': False,  # Quantidade de valores unicos na coluna
        'moda': True,  # Moda (apenas a primeira se houver mais de uma)
        'mean': True,  # Média
        'std': False,  # Desvio padrão

        # Amplitude - É a distancia entre o valor máximo e o mínimo.
        'AMP': False,
        # Desvio Médio Absoluto - É a média dos desvios, ou seja, a média aritmética dos valores absolutos dos desvios.
        'MAD': False,
        'VAR': False,  # Variância - é o quadrado do desvio padrão(std²)
        # Coeficiente de Variação - O coeficiente de variação é uma medida de dispersão relativa que expressa a variabilidade dos dados em relação à média. O coeficiente de variação é uma medida adimensional, ou seja, não possui unidade de medida. Quanto menor for o seu valor, mais homogêneo é o conjunto de dados.
        'CV': False,

        # Quantidade de registros com  valor abaixo de 3 DESVIOS PADRÃO a partir da MÉDIA
        'qtd_3SZ<': False,
        # Quantidade de registros com valor abaixo do PRIMEIRO QUARTIL subtraído do (intervalo inter-quartil(IQR) * 1,5)
        'qtd_LOWER': False,
        'qtd_01%': False,  # Quantidade de registros no PRIMEIRO percentil

        '3SZ<': True,  # Valor de 3 DESVIOS PADRÃO abaixo da MÉDIA
        # Valor do PRIMEIRO QUARTIL subtraído do (intervalo inter-quartil(IQR) * 1,5)
        'LOWER': True,

        'min': True,  # Menor valor encontrado
        '01%': True,  # Valor do PRIMEIRO percentil
        '25%': True,  # PRIMEIRO QUARTIL
        '50%': True,  # SEGUNDO QUARTIL ou MEDIANA
        '75%': True,  # TERCEIRO QUARTIL
        '99%': True,  # Valor do ÚLTIMO percentil
        'max': True,  # Maior valor encontrado

        # Valor do TERCEIRO QUARTIL acrescido do (intervalo inter-quartil(IQR) * 1,5)
        'UPPER': True,
        '3SZ>': True,  # Valor de 3 DESVIOS PADRÃO acima da MÉDIA

        'qtd_99%': False,  # Quantidade de registros no ÚLTIMO percentil
        # Quantidade de registros com valor acima do TERCEIRO QUARTIL acrescido do (intervalo inter-quartil(IQR) * 1,5)
        'qtd_UPPER': False,
        # Quantidade de registros com valor acima de 3 DESVIOS PADRÃO a partir da MÉDIA
        'qtd_3SZ>': False,

        # Assimetria das caldas do gráfico em relação à distribuição normal. Se a assimetria for zero, a distribuição é simétrica. Se a assimetria for positiva, a distribuição é assimétrica à direita, ou seja, a cauda direita é mais longa. Se a assimetria for negativa, a distribuição é assimétrica à esquerda, ou seja, a cauda esquerda é mais longa.
        'skewness': False,
        # Curtose da altura do gráfico em relação à distribuição normal. Se a curtose for zero (considerando a fórmula do Pandas), a distribuição é mesocúrtica, ou seja, tem o mesmo achatamento da distribuição normal. Se a curtose for positiva, a distribuição é leptocúrtica, ou seja, tem um pico mais alto e caudas mais pesadas do que a distribuição normal. Se a curtose for negativa, a distribuição é platicúrtica, ou seja, tem um pico mais baixo e caudas mais leves do que a distribuição normal.
        'kurtosis': False,
        # Indica se segue uma distribuição normal com a partir do teste selecionado
        'dist_normal': False,
        # Indica o valor P do teste selecionado para verficar se a distruibuição segue a normal
        'pvalue': False,
        # Indica o teste selecionado para verficar se a distruibuição segue a normal
        'teste normalidade': False,
    }

################################################################################

    # se não for mostrar e nem retornar, não faz nada
    if flg_mostrar_dataframe == False and flg_retornar_dataframe == False:
        return None

    df_temp_num = df.select_dtypes(
        include='number')  # somente colunas numéricas
    if df_temp_num.empty:  # Não há colunas numéricas no dataframe
        return None

    df_dscrb = df_temp_num.describe().T  # describe() transposto

    df_dscrb['soma'] = df_temp_num.sum()  # Soma total dos valores da coluna
    # Quantidade de valores unicos na coluna
    df_dscrb['unicos'] = df_temp_num.nunique()
    df_dscrb['tipo'] = df_temp_num.dtypes.values  # tipo de dado da coluna
    df_dscrb['percentual inteiros'] = 100.0 * (df_temp_num.astype(float).map(
        # percentual de inteiros na coluna
        float.is_integer).sum()) / df_dscrb['count']

    df_dscrb['nulo'] = df_temp_num.isna().sum()  # quantidade de nulos
    df_dscrb['nulo/count'] = df_dscrb['nulo'] / df_dscrb['count']
    df_dscrb['moda'] = df_temp_num.mode().iloc[0]  # primeira MODA

    # PERCENTIL 1 e 99
    df_dscrb['01%'] = df_temp_num.quantile(q=0.01).values
    df_dscrb['99%'] = df_temp_num.quantile(q=0.99).values

    # IQR
    df_dscrb['LOWER'] = df_dscrb['25%'] - 1.5 * \
        (df_dscrb['75%'] - df_dscrb['25%'])
    df_dscrb['UPPER'] = df_dscrb['75%'] + 1.5 * \
        (df_dscrb['75%'] - df_dscrb['25%'])

    # SCORE Z
    df_dscrb['3SZ<'] = df_dscrb['mean'] - 3 * df_dscrb['std']
    df_dscrb['3SZ>'] = df_dscrb['mean'] + 3 * df_dscrb['std']

    # quantidades relacionadas
    df_dscrb['qtd_01%'] = (df_temp_num < df_dscrb['01%']).sum()
    df_dscrb['qtd_99%'] = (df_temp_num > df_dscrb['99%']).sum()
    df_dscrb['qtd_LOWER'] = (df_temp_num < df_dscrb['LOWER']).sum()
    df_dscrb['qtd_UPPER'] = (df_temp_num > df_dscrb['UPPER']).sum()
    df_dscrb['qtd_3SZ<'] = (df_temp_num < df_dscrb['3SZ<']).sum()
    df_dscrb['qtd_3SZ>'] = (df_temp_num > df_dscrb['3SZ>']).sum()

    # dispersão absoluta e relativa
    df_dscrb['AMP'] = df_temp_num.max() - df_temp_num.min()
    df_dscrb['MAD'] = (df_temp_num - df_temp_num.mean()).abs().mean()
    df_dscrb['VAR'] = df_temp_num.var()
    df_dscrb['CV'] = df_temp_num.std() / df_temp_num.mean() * 100

    # Medidas de forma
    # para uma distribuição ser considerada normal, o valor de skew() deve estar idealmente entre -0.5 e 0.5.
    df_dscrb['skewness'] = df_temp_num.skew()
    # para uma distribuição ser considerada normal (usando kurtosis de Fisher), o valor deve estar próximo de 0.
    df_dscrb['kurtosis'] = df_temp_num.kurtosis()

    '''TESTE NORMALIDADE
        O teste de hipóteses assume as seguintes opções:
            - H_nula: a amostra provém de uma população com distribuição normal - valor p > nivel_significancia
            - H_alternativa: a amostra não provém de uma população com distribuição normal - valor p <= nivel_significancia'''
    lst_testes_normal = []
    lst_testes_pvalue = []
    politica_nulos = 'omit'  # ('propagate', 'omit', 'raise')

    ############################################################################
    ############## ATENCAO A REGRA PARA ESCOLHER SHAPIRO OU KS #################
    ############################################################################
    if teste_normalidade == 'automatico':
        metodo_teste_dist_normal = 'Shapiro-Wilk' if df_temp_num.shape[0] <= 5_000 else 'Kolmogorov-Smirnov'
    elif teste_normalidade == 'Shapiro-Wilk':
        metodo_teste_dist_normal = 'Shapiro-Wilk'
    else:  # Kolmogorov-Smirnov (KS)
        metodo_teste_dist_normal = 'Kolmogorov-Smirnov'

    for i in df_temp_num.columns:
        if metodo_teste_dist_normal == 'Shapiro-Wilk':
            teste_dist_normal = shapiro(  # Teste de Shapiro-Wilk - de 4 a 5_000 amostras/populacao
                x=df_temp_num[i],
                nan_policy=politica_nulos,
            )
        else:
            teste_dist_normal = kstest(  # Teste de Kolmogorov-Smirnov (K-S) - 30 ou mais amostras/populacao
                rvs=df_temp_num[i],
                nan_policy=politica_nulos,
                cdf=norm.cdf,  # Distribuição Normal
                args=(df_temp_num[i].mean(), df_temp_num[i].std()),
            )

        # Nível de significância (alpha) - É a probabilidade de rejeitar a hipótese nula quando ela é verdadeira.
        if teste_dist_normal.pvalue > nivel_significancia:
            # não rejeitamos a hipótese nula... é distribuição normal
            lst_testes_normal.append(True)
        else:
            # rejeitamos a hipótese nula... distribuição não é normal
            lst_testes_normal.append(False)
        lst_testes_pvalue.append(teste_dist_normal.pvalue)

    df_dscrb['dist_normal'] = lst_testes_normal
    df_dscrb['pvalue'] = lst_testes_pvalue
    df_dscrb['teste normalidade'] = metodo_teste_dist_normal

    # reduzindo o tamanho das variáveis numericas
    # somente colunas numéricas
    for coluna in df_dscrb.select_dtypes(include='number').columns:
        # todas se tornam o menor float possível de acordo com o conteúdo
        # apenas as que podem ser inteiros são transformadas para o menor inteiro possível
        # apenas as que podem ser naturais são transformadas para o menor natural possível
        for tipo_base in ('float', 'integer', 'unsigned'):
            df_dscrb[coluna] = pd.to_numeric(
                df_dscrb[coluna], downcast=tipo_base)

    ############################################################################
    ############### ATENCAO AO DICIONARIO NO INICIO DA FUNCAO ##################
    ############################################################################
    if flg_mostrar_dataframe:
        # tira o limite de linhas e colunas a mostrar no notebook
        with pd.option_context('display.max_columns', None, 'display.max_rows', None):
            # formatando o dataframe com o describe de acordo com o dicionário acima
            df_colorido = df_dscrb[dct_colunas_coloridas.keys()] \
                .style.background_gradient(
                    # só colore quem for True no dicionario
                    subset=[k for k, v in dct_colunas_coloridas.items() if v],
                    cmap=palette,  # determina qual será o mapa de cores
                    axis='columns',
            )

            display(df_colorido)

    if flg_retornar_dataframe:
        return df_dscrb


#######################################################################################
# Função que junta os coeficientes de saída de um gridsearch com seus respectivos nomes
#######################################################################################
def fnc_dataframe_coeficientes(
    colunas: list | tuple,  # Nomes das features que saíram do grid_search
    coeficientes: list | tuple,  # Coeficientes das features que saíram do grid_search
    # Nome da coluna que conterá os coeficientes
    nome_coluna_coeficientes: str = 'coeficiente',
    # Define se a ordenação dos coeficientes será ascendente
    flg_ordencao_ascendente: bool = False,
    # Define se a ordenação será pelo valor absoluto dos coeficientes
    flg_ordenacao_por_modulo: bool = True,
    # Se True, retorna o dataframe para uso posterior
    flg_retornar_dataframe: bool = True,
    flg_mostrar_dataframe: bool = True,  # Se True, exibe o dataframe colorido
    # Paleta de cores utilizada para o dataframe colorido
    paleta_de_cores=PALETTE_TEMPERATURA,
    precisao: int = None,  # Número de casas decimais a serem exibidas
    flg_horizontal: bool = False,  # Se True, transpõe o dataframe para uma única linha
) -> pd.DataFrame:  # Retorna um dataframe com os coeficientes identificados
    '''
        Combina os coeficientes de um gridsearch com os nomes das features em um DataFrame.

        Args:
            colunas (list | tuple): Nomes das features resultantes do grid_search.
            coeficientes (list | tuple): Coeficientes correspondentes às features.
            nome_coluna_coeficientes (str, opcional): Nome da coluna que armazenará os coeficientes (default: 'coeficiente').
            flg_ordencao_ascendente (bool, opcional): Define a ordem de retorno dos coeficientes (default: False).
            flg_ordenacao_por_modulo (bool, opcional): Se True, ordena pelo valor absoluto dos coeficientes (default: True).
            flg_retornar_dataframe (bool, opcional): Se True, retorna o dataframe (default: True).
            flg_mostrar_dataframe (bool, opcional): Se True, exibe o dataframe colorido (default: True).
            paleta_de_cores (list, opcional): Paleta de cores para o dataframe colorido (default: PALETTE_TEMPERATURA).
            precisao (int, opcional): Número de casas decimais a serem exibidas (default: None).
            flg_horizontal (bool, opcional): Se True, mostra os coeficientes em uma única linha (default: False).

        Returns:
            pd.DataFrame: DataFrame com os coeficientes e suas respectivas features, ordenados conforme os parâmetros especificados.

        Examples:
            >>> df_coeficientes = fnc_dataframe_coeficientes(
                    colunas=grid_search.best_estimator_.regressor_['preprocessor'].get_feature_names_out(),
                    coeficientes=grid_search.best_estimator_.regressor_['regressor'].coef_,
                )
            >>> display(df_coeficientes)
    '''

    # Cria um DataFrame com os coeficientes e os nomes das colunas
    df_coeficientes = pd.DataFrame(
        data=coeficientes,
        index=colunas,
        columns=[nome_coluna_coeficientes]
    )

    # Ordena o DataFrame conforme a escolha do usuário
    if flg_ordenacao_por_modulo:
        # Cria coluna temporária de valores absolutos
        df_coeficientes['modulo_temporario'] = df_coeficientes[nome_coluna_coeficientes].abs()
        df_coeficientes.sort_values(
            by='modulo_temporario', ascending=flg_ordencao_ascendente, inplace=True)
        del df_coeficientes['modulo_temporario']  # Remove a coluna temporária
    else:
        df_coeficientes.sort_values(
            by=nome_coluna_coeficientes, ascending=flg_ordencao_ascendente, inplace=True)

    # Transpõe o DataFrame se necessário
    if flg_horizontal:
        df_coeficientes = df_coeficientes.T
        eixo = 'columns'
    else:
        eixo = 'index'

    # Exibe o DataFrame colorido se solicitado
    if flg_mostrar_dataframe:
        display(df_coeficientes.style.background_gradient(
            cmap=paleta_de_cores, axis=eixo).format(precision=precisao))

    # Retorna o DataFrame se solicitado
    if flg_retornar_dataframe:
        return df_coeficientes


###############################################################################
# Carrega os dados do Kaggle num DataFrame do pandas
###############################################################################
def fnc_carregar_dados_kaggle(kaggle_handle: str, file_path: str) -> pd.DataFrame:
    '''
        Carrega os dados do Kaggle usando o KaggleDatasetAdapter e retorna um DataFrame do pandas.

        Args:
            kaggle_handle (str): O caminho para o dataset no Kaggle.
            file_path (str): O arquivo que você quer carregar do kaggle.

        Returns:
            pd.DataFrame: O DataFrame do pandas com os dados carregados.
    '''
    # Carregue a versão mais recente do Kaggle
    df = load_dataset(
        adapter=KaggleDatasetAdapter.PANDAS,
        handle=kaggle_handle,
        path=file_path,
    )

    return df

###############################################################################
