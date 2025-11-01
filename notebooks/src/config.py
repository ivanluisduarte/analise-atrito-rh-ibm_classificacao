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

import warnings
from pathlib import Path

warnings.filterwarnings('ignore')  # para ignorar warnings desnecessários

PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / 'dados'

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_LIMPOS = PASTA_DADOS / 'ibm_hr_attrition.parquet'

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
PASTA_MODELOS = PASTA_PROJETO / 'modelos'

# modelo final treinado
MODELO_FINAL = PASTA_MODELOS / 'modelo_ibm_hr_attrition.joblib'

# coloque abaixo outros caminhos que você julgar necessário
PASTA_RELATORIOS = PASTA_PROJETO / 'relatorios'
PASTA_IMAGENS = PASTA_RELATORIOS / 'imagens'

N_JOBS = -2  # número de job/processadores simultaneos, -2 deixa 1 núcleo livre
RANDOM_STATE = 123  # semente para reprodução dos resultados


PASTA_CONFIGURACAO = PASTA_PROJETO / 'config'

# dicionário para configuração das colunas
DCT_CONFIGURACAO_COLUNAS = PASTA_CONFIGURACAO / 'dct_configuracao_colunas.json'
