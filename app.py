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


########################################################################################################################
# %% IMPORTAÇÕES

import streamlit as st  # interface WEB - https://streamlit.io/

########################################################################################################################
# %% FUNÇÕES CACHE_DATA
# qualquer coisa que possa ser armazenado em database
# Python primitives, dataframe e API calls


@st.cache_data
def fnc_dct_help() -> dict:
    '''
    Retorna um dicionário com descrições (ajuda) das colunas exibidas na UI.

    Essas descrições são usadas como textos de ajuda (tooltips) nos componentes do
    Streamlit para orientar o usuário sobre o significado de cada campo.

    Somente os campos exibidos na UI estão incluídos neste dicionário.

    Returns:
        dict: mapeamento coluna -> descrição (str).
    '''
    return {
        # 'Age': 'Idade do funcionário (valor numérico).',
        # 'Attrition': 'Se o funcionário deixou a empresa (Yes/No).',
        'BusinessTravel': 'Frequência de viagens a negócios',
        'DailyRate': 'Taxa diária (taxa por dia para a empresa).',
        # 'Department': 'Departamento.',
        'Department-JobRole': 'Departamento e Função no trabalho.',
        # 'DistanceFromHome': 'Distância da casa para o trabalho (em unidades, valor numérico).',
        # 'Education': 'Nível de educação.',
        # 'EducationField': 'Área de formação educacional.',
        # 'EmployeeCount': 'Contagem de funcionários.',
        # 'EmployeeNumber': 'Número do funcionário.',
        'EnvironmentSatisfaction': 'Satisfação com o ambiente de trabalho.',
        # 'Gender': 'Gênero/sexo.',
        # 'HourlyRate': 'Taxa horária (taxa por hora para a empresa).',
        # 'JobInvolvement': 'Envolvimento no trabalho.',
        # 'JobLevel': 'Nível do cargo.',
        # 'JobRole': 'Função no trabalho.',
        'JobSatisfaction': 'Satisfação com o trabalho.',
        'MaritalStatus': 'Estado civil.',
        # 'MonthlyIncome': 'Renda mensal/Salário.',
        # 'MonthlyRate': 'Taxa mensal (taxa por mês para a empresa).',
        'NumCompaniesWorked': 'Número de empresas em que trabalhou anteriormente.',
        # 'Over18': 'Se o funcionário tem mais de 18 anos.',
        'OverTime': 'Se trabalha horas extras.',
        # 'PercentSalaryHike': 'Percentual de aumento salarial.',
        # 'PerformanceRating': 'Avaliação de desempenho.',
        # 'RelationshipSatisfaction': 'Satisfação com relacionamentos no trabalho.',
        # 'StandardHours': 'Horas padrão de trabalho.',
        'StockOptionLevel': 'Nível de opções de ações.',
        'TotalWorkingYears': 'Total de anos de experiência profissional.',
        # 'TrainingTimesLastYear': 'Número de treinamentos no último ano.',
        'WorkLifeBalance': 'Equilíbrio entre trabalho e vida pessoal.',
        'YearsAtCompany': 'Anos na empresa atual.',
        'YearsInCurrentRole': 'Anos no cargo atual.',
        'YearsSinceLastPromotion': 'Anos desde a última promoção.',
        'YearsWithCurrManager': 'Anos com o gerente atual.',
    }


@st.cache_data
def fnc_carregar_dct_configuracao_colunas() -> dict:
    '''
    Carrega o arquivo JSON de configuração das colunas (labels, categorias, máximo, mínimo e moda).

    A função faz a leitura do caminho definido em notebooks.src.config.DCT_CONFIGURACAO_COLUNAS
    e retorna o conteúdo decodificado como um dicionário Python.

    Returns:
        dict: configuração das colunas utilizada para montar os widgets da UI.
    '''
    # importação local para evitar cache desnecessário
    from json import load

    from notebooks.src.config import DCT_CONFIGURACAO_COLUNAS

    with open(DCT_CONFIGURACAO_COLUNAS, 'r') as f:
        return load(f)


########################################################################################################################
# %% FUNÇÕES CACHE_RESOURCE
# qualquer coisa que NÃO possa ser armazenado em database
# ML models e database connections

@st.cache_resource
def fnc_carregar_modelo():
    '''
    Carrega o modelo treinado a partir do arquivo definido em notebooks.src.config.MODELO_FINAL.

    Esta função é decorada com st.cache_resource para manter o modelo em memória entre execuções,
    evitando recarregamentos repetidos e melhorando a performance da aplicação.

    Returns:
        objeto do modelo: modelo(imblearn.pipeline.Pipeline) carregado via joblib.load para predição.
    '''
    # importação local para evitar cache desnecessário
    from joblib import load
    from notebooks.src.config import MODELO_FINAL

    return load(MODELO_FINAL)


########################################################################################################################
# %% carregando arquivos ou seu cache, se já existir
dct_help = fnc_dct_help()
dct_configuracao_colunas = fnc_carregar_dct_configuracao_colunas()

modelo = fnc_carregar_modelo()


# dicionário vazio para armazenar as respostas do formulário
dct_respostas = {}

# lista de campos de entrada
lst_campos = sorted(dct_help.keys())

########################################################################################################################
# %% FUNÇÕES AUXILIARES


def fnc_selectbox_categorica(coluna: str) -> str:
    '''
    Renderiza um selectbox (menu suspenso) para uma coluna categórica e retorna o valor selecionado.

    O label, as opções e o valor padrão (mode) são obtidos a partir da configuração carregada.
    A ajuda exibida no componente é retirada do dicionário dct_help.

    Args:
        coluna (str): nome da coluna categórica a ser exibida.

    Returns:
        str: valor selecionado pelo usuário.
    '''
    configuracoes = dct_configuracao_colunas['colunas_categoricas'][coluna]
    categories = configuracoes['categories']

    return st.selectbox(
        label=configuracoes['label'],
        options=categories,
        index=categories.index(configuracoes['mode']),
        help=dct_help[coluna],
    )


def fnc_slider_numerica(coluna: str) -> int:
    '''
    Renderiza um slider para uma coluna numérica e retorna o valor selecionado.

    Os limites, o valor padrão e o rótulo são lidos da configuração de colunas.

    Args:
        coluna (str): nome da coluna numérica a ser exibida.

    Returns:
        int: valor inteiro selecionado no slider.
    '''
    configuracoes = dct_configuracao_colunas['colunas_numericas'][coluna]

    return st.slider(
        label=configuracoes['label'],
        min_value=configuracoes['min'],
        max_value=configuracoes['max'],
        value=configuracoes['mode'],
        step=1,
        format='%d',
        help=dct_help[coluna],
    )


def fnc_escolher_objeto(coluna: str) -> None:
    '''
    Decide qual widget deve ser renderizado para a coluna informada e grava a resposta.

    Se a coluna estiver entre as categóricas, usa um selectbox; caso contrário,
    usa um slider. O valor escolhido é salvo no dicionário global dct_respostas.

    Args:
        coluna (str): nome da coluna a ser exibida e cujo valor será coletado.
    '''
    if coluna in dct_configuracao_colunas['colunas_categoricas'].keys():
        dct_respostas[coluna] = fnc_selectbox_categorica(coluna)
    else:
        dct_respostas[coluna] = fnc_slider_numerica(coluna)


########################################################################################################################
# %% CONFIGURAÇÕES DE PÁGINA
titulo = 'Análise de atrito de funcionários - RH IBM'

st.set_page_config(
    page_title=titulo,
    # https://share.streamlit.io/streamlit/emoji-shortcodes
    page_icon=':office:',
    layout='wide',
)


########################################################################################################################
# %% PAGINA WEB

col_titulo, col_github = st.columns(spec=(0.8, 0.2))

with col_titulo:
    st.title(
        body=titulo,
        help='''Baseado em informações sobre funcionários de uma empresa fictícia,
como idade, gênero, estado civil, nível de satisfação, entre outros. O objetivo
é prever a probabilidade de um funcionário sair da empresa.

O objetivo deste trabalho é apenas para compor o portfólio de ciência de dados de Ivan Luís Duarte.

Projeto no GitHub: https://github.com/ivanluisduarte/analise-atrito-rh-ibm_classificacao

GitHub: https://github.com/ivanluisduarte

LinkedIn: https://www.linkedin.com/in/ivanluisduarte/
''',
    )

with col_github:
    st.markdown(
        body='''
        <div>
            <a href="https://github.com/ivanluisduarte/analise-atrito-rh-ibm_classificacao">
                🚀 github.com/ivanluisduarte
            </a>
        </div>
        ''',
        unsafe_allow_html=True,
        help='Link para o repositório do projeto no GitHub.',
    )

# st.markdown(body='---')  # Linha horizontal

########################################################################################################################
# %% FORMULÁRIO DE ENTRADA DE DADOS


with st.form(
    key='formulario',
    clear_on_submit=False,
    border=False,
):

    # dividindo a tela em 3 colunas - formulário
    coluna_1, coluna_2, coluna_3 = st.columns(
        spec=(0.38, 0.31, 0.31),
        gap='small',
        border=True,
    )

    with coluna_1:
        for i in range(0, 5):
            fnc_escolher_objeto(lst_campos[i])

    with coluna_2:
        for i in range(5, 10):
            fnc_escolher_objeto(lst_campos[i])

    with coluna_3:
        for i in range(10, 15):
            fnc_escolher_objeto(lst_campos[i])

    # st.markdown(body='---')  # Linha horizontal


########################################################################################################################
# PREVISÃO E EXIBIÇÃO DO RESULTADO

    with st.container(border=False):
        col_botao, col_metrica, col_texto = st.columns(spec=(0.15, 0.15, 0.7))

        with col_botao:
            flg_confirmou = st.form_submit_button(
                label='Prever Atrito',
                use_container_width=True,
                help='''Ao clicar nesse botão, uma previsão será feita para a
probabilidade de um funcionário sair da empresa com base nas informações
fornecidas.''',
            )

        if flg_confirmou:
            # importação local para evitar cache desnecessário
            from pandas import DataFrame

            # criando o dataframe para predição
            df = DataFrame([dct_respostas])

            # fazendo a predição com o modelo carregado
            flg_attrition = modelo.predict(df)[0]
            prob_attrition = round(modelo.predict_proba(df)[0][flg_attrition] *
                                   100.0, 2)  # probabilidade em %

            # exibindo o resultado na tela
            if flg_attrition:
                value = '\U0001f534 Yes'
                delta_color = 'inverse'
                cor = 'red'
                diagnostico = f'''O modelo previu que o funcionário tem
{prob_attrition:.2f}% de chance de sair da empresa.'''
                prescricao = '''Considere implementar estratégias de retenção
para esse funcionário.'''

            else:
                value = '\U0001f7e2 No'
                delta_color = 'normal'
                cor = 'green'
                diagnostico = f'''O modelo previu que o funcionário tem
{prob_attrition:.2f}% de chance de ficar na empresa.'''
                prescricao = '''Continue investindo em estratégias de retenção
para manter esse funcionário satisfeito.'''

            with col_metrica:
                st.metric(
                    label='Predição de Atrito:',
                    value=value,
                    delta=prob_attrition,
                    delta_color=delta_color,
                    help='''Indica se o funcionário provavelmente sairá (Yes)
ou permanecerá (No) na empresa.  \nO valor representa o percentual de
confiança associada a essa predição.''',
                )

                with col_texto:
                    st.markdown(
                        body=F'''<span style="color:{cor}; font-weight:bold;">
                            {diagnostico}</span><br>{prescricao}''',
                        unsafe_allow_html=True
                    )

                ########################################################################################################################
