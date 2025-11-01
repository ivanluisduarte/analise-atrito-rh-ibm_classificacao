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

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from IPython.display import display
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.preprocessing import label_binarize

from .config import N_JOBS, RANDOM_STATE
from .graficos import PALETTE_TEMPERATURA


###############################################################################
# Função que retorna um pipeline de modelo pronto para treinamento
###############################################################################
def fnc_construir_pipeline_modelo_classificacao(
    # PARAMETRO OBRIGATORIO
    classificador: ClassifierMixin,

    # PARAMETROS OPCIONAIS
    preprocessor: BaseEstimator = None,
    sampler: BaseSampler = None,

    nome_preprocessor: str = 'preprocessor',
    nome_sampler: str = 'sampler',
    nome_classificador: str = 'clf',
) -> TransformedTargetRegressor:
    """
    Função que retorna um pipeline de modelo pronto para treinamento.

    Args:
        classificador (ClassifierMixin): Classe de classificador que deve implementar os métodos fit e predict.
        preprocessor (BaseEstimator, optional): Classe de préprocessamento de features que deve implementar fit e transform. Padrão  None.
        sampler (BaseSampler, optional): Classe de amostragem de dados que deve implementar fit e sample. Padrão None.
        nome_preprocessor (str, optional): Nome que aparecer na saida identificando o pr -processador. Padrão  'preprocessor'.
        nome_sampler (str, optional): Nome que aparecer na saida identificando a amostragem de dados. Padrão  'sampler'.
        nome_classificador (str, optional): Nome que aparecer na saida identificando o classificador. Padrão  'clf'.

    Returns:
        TransformedTargetClassifier: Modelo configurado pronto para treinamento.
    """
    lst_etapas = []

    if preprocessor:
        lst_etapas.append((nome_preprocessor, preprocessor))

    if sampler:
        lst_etapas.append((nome_sampler, sampler))

    lst_etapas.append((nome_classificador, classificador))

    return Pipeline(lst_etapas)


############################################################################################
# Função para treinar modelos de classificação com validação cruzada e avaliar o desempenho.
############################################################################################
def fnc_treinar_e_validar_modelo_classificacao(
    X: pd.DataFrame | pd.Series,  # Dados preditores
    y: pd.DataFrame | pd.Series,  # Variável predita
    # Dicionário de modelos com pré-processadores, amostragem de dados e classificadores
    classificadores: dict,
    nome_preprocessor: str = 'preprocessor',
    nome_sampler: str = 'sampler',
    nome_classificador: str = 'clf',
    n_splits: int = 5,  # Número de folds na validação cruzada
    shuffle: bool = True,  # Se True, embaralha os dados antes da divisão
    random_state: int = RANDOM_STATE,  # Semente para reprodutibilidade
    scoring: list | tuple = (  # lista/tupla com métricas de avaliação
        # Proporção de acertos corretos (predições corretas / total de amostras).
        # 'accuracy',

        # A média da recall (sensibilidade) calculada para cada classe.
        # Mais robusta que 'accuracy' em casos de classes desbalanceadas.
        'balanced_accuracy',

        # Proporção de acertos corretos onde o rótulo verdadeiro está entre os K principais rótulos preditos (para modelos que fornecem probabilidades).
        # 'top_k_accuracy',

        # Área sob a curva de Precisão-Recall (Precision-Recall Curve).
        # Útil para dados desbalanceados ou quando o custo de Falsos Positivos e Falsos Negativos é diferente.
        'average_precision',

        # Brier Score Negativo. O Brier Score mede o erro quadrático médio das probabilidades preditas.
        # É uma medida de "loss" (perda), então o valor é negado para que scores maiores sejam melhores.
        # 'neg_brier_score',

        # Média harmônica da precisão (Precision) e recall (Recall).
        # Bom para problemas onde Falsos Positivos e Falsos Negativos têm custos semelhantes e o desbalanceamento de classes é um problema.
        'f1',

        # F1 score calculado globalmente contando o total de verdadeiros positivos, falsos negativos e falsos positivos.
        # Útil quando se quer ignorar o desbalanceamento de classes e trata-se de um problema multiclasse.
        # 'f1_micro',

        # F1 score calculado para cada classe, e a média não ponderada (igual peso) desses scores é retornada.
        # Trata todas as classes igualmente.
        # 'f1_macro',

        # F1 score calculado para cada classe, e a média ponderada pelo número de amostras em cada classe é retornada.
        # Leva em conta o desbalanceamento de classes.
        # 'f1_weighted',

        # F1 score para classificação multi-rótulo (multilabel), calculado por amostra e depois tirada a média.
        # 'f1_samples',

        # Negative Log-Loss (também conhecida como Cross-Entropy Loss). Mede a incerteza das probabilidades do modelo.
        # É uma medida de "loss" (perda), então é negada para que scores maiores sejam melhores (o log-loss real é menor=melhor).
        # 'neg_log_loss',

        # Proporção de predições positivas que foram, de fato, corretas (TP / (TP + FP)).
        # Aborda a questão: Das vezes que o modelo previu 'Positivo', quantas estavam corretas?
        'precision',

        # Proporção de amostras positivas que foram identificadas corretamente (TP / (TP + FN)).
        # Aborda a questão: De todas as amostras que são realmente 'Positivas', quantas o modelo identificou?
        'recall',

        # Proporção de acertos corretos (união das amostras corretamente preditas e verdadeiras, dividido pela união das amostras preditas e verdadeiras).
        # Semelhante ao F1, mas para a intersecção/união. Para multi-rótulo/multiclasse.
        # 'jaccard',

        # Área Sob a Curva ROC (Receiver Operating Characteristic).
        # Mede a capacidade do modelo de distinguir entre classes.
        # Requer probabilidades preditas ou pontuações de decisão.
        'roc_auc',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-Rest" (OVR) e média "macro" (não ponderada).
        # 'roc_auc_ovr',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-One" (OVO) e média "macro" (não ponderada).
        # 'roc_auc_ovo',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-Rest" (OVR) e média "weighted" (ponderada pelo suporte de cada classe).
        # 'roc_auc_ovr_weighted',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-One" (OVO) e média "weighted" (ponderada pelo suporte de cada classe).
        # 'roc_auc_ovo_weighted',

        # Métrica de "D2 score" baseada no Log Loss. Mede a proporção de desvio ("deviance") log-loss explicado pelo modelo.
        # 'd2_log_loss_score',
    ),
) -> tuple:  # DataFrame com resultados da validação cruzada
    """
    Função para treinar modelos de classificação com validação cruzada e avaliar o desempenho.

    Args:
        X (pd.DataFrame | pd.Series): Dados preditores
        y (pd.DataFrame | pd.Series): Variável predita
        classificadores (dict): Dicionário de models com pré-processadores, amostragem de dados e classificadores
        nome_preprocessor (str, optional): Nome que aparecer na saida identificando o pré-processador. Padrão  'preprocessor'.
        nome_sampler (str, optional): Nome que aparecer na saida identificando a amostragem de dados. Padrão  'sampler'.
        nome_classificador (str, optional): Nome que aparecer na saida identificando o classificador. Padrão  'clf'.
        n_splits (int, optional): Número de folds na validação cruzada. Padrão 5.
        shuffle (bool, optional): Se True, embaralha os dados antes da divisão. Padrão True.
        random_state (int, optional): Semente para reprodutibilidade. Padrão RANDOM_STATE.
        scoring (list | tuple, optional): Lista/tupla com métricas de avaliação. Padrão ('accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'average_precision').

    Returns:
        pd.DataFrame: DataFrame com resultados da validação cruzada
    """
    # Configura a validação cruzada KFold
    kf = KFold(
        n_splits=n_splits,  # Número de folds
        shuffle=shuffle,  # Se True, embaralha os dados antes da divisão
        random_state=random_state  # Semente para reprodutibilidade
    )

    resultados = {}  # Dicionário para armazenar os resultados de cada modelo
    pipelines = {}  # Dicionário para armazenar os modelos sem treinamento

    for nome_modelo, clf_config in classificadores.items():  # Itera sobre os modelos
        # Constrói o pipeline do modelo
        pipelines[nome_modelo] = fnc_construir_pipeline_modelo_classificacao(
            classificador=clf_config[nome_classificador],
            preprocessor=clf_config.get(nome_preprocessor, None),
            sampler=clf_config.get(nome_sampler, None),
            nome_preprocessor=nome_preprocessor,
            nome_sampler=nome_sampler,
            nome_classificador=nome_classificador,
        )

        # Realiza a validação cruzada
        resultados[nome_modelo] = cross_validate(
            estimator=pipelines[nome_modelo],  # Modelo a ser treinado
            X=X,  # Dados preditores
            y=y,  # Variável predita
            cv=kf,  # Configuração da validação cruzada
            scoring=scoring,  # Métricas de avaliação
        )

    # Organiza os resultados em um DataFrame e retorna
    return (fnc_organiza_resultados(resultados), pipelines)


#########################################################################################################################
# função para organizar os resultados da validação cruzada feita pela função treinar_e_validar_modelo_regressao() acima #
#########################################################################################################################
def fnc_organiza_resultados(
    dct_resultados: dict,  # Dicionário com os resultados da validação cruzada
) -> pd.DataFrame:  # DataFrame com os resultados organizados
    '''
    Função para organizar os resultados da validação cruzada em um DataFrame.

    Args:
        dct_resultados (dict): Dicionário contendo os resultados da validação cruzada. Cada chave deve representar um modelo e cada valor deve ser outro dicionário com os resultados correspondentes, incluindo 'fit_time' e 'score_time'.

    Returns:
        pd.DataFrame: Um DataFrame contendo os resultados organizados da validação cruzada, onde cada linha representa uma iteração de validação e as colunas incluem os resultados e o tempo total de treinamento e validação.

    Examples:
        >>> df_resultados = fnc_organiza_resultados(dct_resultados)
        >>> print(df_resultados.head())

    Notes:
        O DataFrame resultante terá uma coluna 'model' com os nomes dos modelos e as colunas de resultados expandidas, além de uma coluna adicional 'time_seconds' que representa o tempo total gasto em cada iteração de validação cruzada.
    '''

    # Calcula o tempo total de treinamento e validação para cada modelo
    for chave in dct_resultados:
        dct_resultados[chave]['time_seconds'] = (
            dct_resultados[chave]['fit_time'] +
            dct_resultados[chave]['score_time']
        )

    # Cria um DataFrame a partir dos resultados da validação cruzada
    df_resultados = (
        pd.DataFrame(dct_resultados).T.reset_index().rename(
            columns={'index': 'model'})
    )

    # Expande o DataFrame para que cada linha corresponda a uma iteração de validação cruzada
    df_resultados = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    # Converte as colunas para o tipo numérico, ignorando erros de conversão
    for coluna in df_resultados.columns.difference(['model']):
        # reduzindo o tamanho das variáveis numericas
        # todas se tornam o menor float possível de acordo com o conteúdo
        # apenas as que podem ser inteiras são transformadas para o menor inteiro possível
        # apenas as que podem ser naturais são transformadas para o menor natural possível
        for tipo_base in ('float', 'signed', 'unsigned'):
            try:
                df_resultados[coluna] = pd.to_numeric(
                    df_resultados[coluna], downcast=tipo_base)
            except ValueError:
                pass

    lst_tempos = ['time_seconds', 'fit_time', 'score_time']
    lst_demais = [x for x in df_resultados.columns.to_list()
                  if x not in lst_tempos]

    return df_resultados.reindex(columns=lst_demais + lst_tempos)


###############################################################################
# função para planejar modelos de regressão validação cruzada e grid search
###############################################################################
def fnc_grid_search_cv_classificador(
    # PARAMETROS OBRIGATÓRIOS
    # Classe de classificação que deve implementar fit e predict
    classificador: ClassifierMixin,
    # dicionário, ou lista de dicionários contendo os hiperparametros a testar
    param_grid: dict | list,

    # PARAMETROS OPCIONAIS
    # Classe de pré-processamento, deve implementar fit e transform
    preprocessor: BaseEstimator = None,
    # Classe de amostragem de dados, deve implementar fit e sample
    sampler: BaseSampler = None,

    # nome/chave que aparecerá na saída identificando o preprocessor
    nome_preprocessor: str = 'preprocessor',
    # nome/chave que aparecerá na saída identificando o sampler
    nome_sampler: str = 'sampler',
    # nome/chave que aparecerá na saída identificando o classificador
    nome_classificador: str = 'clf',

    n_splits: int = 5,  # numero de pedaços e vezes em que a base será divida para treino
    # True indica que os splits devem ser pegos formados de forma aleatória
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,  # semente para reprodução dos resultados

    # número de job/processadores simultaneos. Se None, 1 job por vez apenas,
    n_jobs: int = N_JOBS,
    # se -1 usa todos os processadores disponíveis, se -2 usa todos -1 (use se a máquina estiver travando usando todos)
    return_train_score: bool = False,  # True se quiser dados de treinamento na saída
    verbose: int = 1,  # nível de log que deseja ver - 1, 2 ou 3
    refit_metric: str = 'f1',  # metrica usada para refit no grid search
    scoring: list | tuple = (  # lista/tupla com métricas de avaliação
        # Proporção de acertos corretos (predições corretas / total de amostras).
        # 'accuracy',

        # A média da recall (sensibilidade) calculada para cada classe.
        # Mais robusta que 'accuracy' em casos de classes desbalanceadas.
        'balanced_accuracy',

        # Proporção de acertos corretos onde o rótulo verdadeiro está entre os K principais rótulos preditos (para modelos que fornecem probabilidades).
        # 'top_k_accuracy',

        # Área sob a curva de Precisão-Recall (Precision-Recall Curve).
        # Útil para dados desbalanceados ou quando o custo de Falsos Positivos e Falsos Negativos é diferente.
        'average_precision',

        # Brier Score Negativo. O Brier Score mede o erro quadrático médio das probabilidades preditas.
        # É uma medida de "loss" (perda), então o valor é negado para que scores maiores sejam melhores.
        # 'neg_brier_score',

        # Média harmônica da precisão (Precision) e recall (Recall).
        # Bom para problemas onde Falsos Positivos e Falsos Negativos têm custos semelhantes e o desbalanceamento de classes é um problema.
        'f1',

        # F1 score calculado globalmente contando o total de verdadeiros positivos, falsos negativos e falsos positivos.
        # Útil quando se quer ignorar o desbalanceamento de classes e trata-se de um problema multiclasse.
        # 'f1_micro',

        # F1 score calculado para cada classe, e a média não ponderada (igual peso) desses scores é retornada.
        # Trata todas as classes igualmente.
        # 'f1_macro',

        # F1 score calculado para cada classe, e a média ponderada pelo número de amostras em cada classe é retornada.
        # Leva em conta o desbalanceamento de classes.
        # 'f1_weighted',

        # F1 score para classificação multi-rótulo (multilabel), calculado por amostra e depois tirada a média.
        # 'f1_samples',

        # Negative Log-Loss (também conhecida como Cross-Entropy Loss). Mede a incerteza das probabilidades do modelo.
        # É uma medida de "loss" (perda), então é negada para que scores maiores sejam melhores (o log-loss real é menor=melhor).
        # 'neg_log_loss',

        # Proporção de predições positivas que foram, de fato, corretas (TP / (TP + FP)).
        # Aborda a questão: Das vezes que o modelo previu 'Positivo', quantas estavam corretas?
        'precision',

        # Proporção de amostras positivas que foram identificadas corretamente (TP / (TP + FN)).
        # Aborda a questão: De todas as amostras que são realmente 'Positivas', quantas o modelo identificou?
        'recall',

        # Proporção de acertos corretos (união das amostras corretamente preditas e verdadeiras, dividido pela união das amostras preditas e verdadeiras).
        # Semelhante ao F1, mas para a intersecção/união. Para multi-rótulo/multiclasse.
        # 'jaccard',

        # Área Sob a Curva ROC (Receiver Operating Characteristic).
        # Mede a capacidade do modelo de distinguir entre classes.
        # Requer probabilidades preditas ou pontuações de decisão.
        'roc_auc',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-Rest" (OVR) e média "macro" (não ponderada).
        # 'roc_auc_ovr',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-One" (OVO) e média "macro" (não ponderada).
        # 'roc_auc_ovo',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-Rest" (OVR) e média "weighted" (ponderada pelo suporte de cada classe).
        # 'roc_auc_ovr_weighted',

        # AUC ROC para problemas multiclasse usando a estratégia "One-vs-One" (OVO) e média "weighted" (ponderada pelo suporte de cada classe).
        # 'roc_auc_ovo_weighted',

        # Métrica de "D2 score" baseada no Log Loss. Mede a proporção de desvio ("deviance") log-loss explicado pelo modelo.
        # 'd2_log_loss_score',
    ),
):
    """
    Função para planejar modelos de classificação com validação cruzada e grid search.

    Args:
        classificador (ClassifierMixin): Classe de classificação que deve implementar fit e predict.
        param_grid (dict | list): dicionário, ou lista de dicionários contendo os hiperparametros a testar.
        preprocessor (BaseEstimator, optional): Classe de pré-processamento, deve implementar fit e transform. Padrão None.
        sampler (BaseSampler, optional): Classe de amostragem de dados, deve implementar fit e sample. Padrão None.
        nome_preprocessor (str, optional): nome/chave que aparecerá na saída identificando o preprocessor. Padrão 'preprocessor'.
        nome_sampler (str, optional): nome/chave que aparecerá na saída identificando o sampler. Padrão 'sampler'.
        nome_classificador (str, optional): nome/chave que aparecerá na saída identificando o classificador. Padrão 'clf'.
        n_splits (int, optional): Número de folds na validação cruzada. Padrão 5.
        shuffle (bool, optional): Se True, embaralha os dados antes da divisão. Padrão True.
        random_state (int, optional): Semente para reprodutibilidade. Padrão RANDOM_STATE.
        n_jobs (int, optional): Número de job/processadores simultaneos. Se None, 1 job por vez apenas, 
            se -1 usa todos os processadores disponíveis, se -2 usa todos -1 (use se a máquina estiver travando usando todos).
        return_train_score (bool, optional): True se quiser dados de treinamento na saída.
        verbose (int, optional): Nível de log que deseja ver - 1, 2 ou 3.
        refit_metric (str, optional): Métrica usada para refit no grid search. Padrão 'f1'.
        scoring (list | tuple, optional): Lista/tupla com métricas de avaliação.

    Returns:
        GridSearchCV: Modelo configurado pronto para treinamento.
    """
    model = fnc_construir_pipeline_modelo_classificacao(
        classificador=classificador,
        preprocessor=preprocessor,
        sampler=sampler,
        nome_preprocessor=nome_preprocessor,
        nome_sampler=nome_sampler,
        nome_classificador=nome_classificador
    )

    # Configura a validação cruzada KFold
    kf = KFold(
        n_splits=n_splits,  # Número de folds
        shuffle=shuffle,  # Se True, embaralha os dados antes da divisão
        random_state=random_state  # Semente para reprodutibilidade
    )

    grid_search = GridSearchCV(
        estimator=model,
        cv=kf,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        verbose=verbose,
    )

    return grid_search


###############################################################################
# Calcula ROC AUC e Average Precision para uma classe ou média
###############################################################################
def fnc_compute_roc_auc_and_avg_precision(
    y_true_bin,
    y_score,
    average='macro',
    multi_class='ovr',
):
    '''
        Calcula ROC AUC e Average Precision para uma classe ou média.

        Parâmetros:
        - y_true_bin: Array-like, rótulos binarizados.
        - y_score: Array-like, pontuações ou probabilidades.
        - average: Tipo de média ('micro', 'samples', 'weighted', 'macro' ou None). None='macro'.
        - multi_class: Configuração erro multiclasse ('raise', 'ovr', 'ovo' ou None). None='ovr'.

        Retorna:
        - Tuple (roc_auc, precision_recall).
    '''

    try:
        roc_auc = roc_auc_score(y_true_bin, y_score,
                                average=average, multi_class=multi_class)
        precision_recall = average_precision_score(
            y_true_bin, y_score, average=average)

        return (roc_auc, precision_recall)
    except:
        return (np.nan, np.nan)
    '''
        Curva ROC
        A curva ROC (Receiver Operating Characteristic) é uma curva que mostra
            a taxa de verdadeiros positivos (recall) em função da taxa de falsos positivos.
        A curva ROC é útil para avaliar a qualidade de um classificador binário em diferentes thresholds.
        A área sob a curva ROC (AUC - Area Under the Curve) é uma métrica que varia entre 0.0 e 1.0,
            onde 0.0 significa um classificador ruim e 1.0 significa um classificador perfeito.    

        Curva Precision-Recall
        A curva precision-recall é uma curva que mostra a precisão em função do recall.
        A curva precision-recall é útil para avaliar a qualidade de um
            classificador binário em diferentes thresholds. 
        A área sob a curva precision-recall (AUC - Area Under the Curve) é uma 
            métrica que varia entre 0.0 e 1.0, onde 0.0 significa um classificador ruim
            e 1.0 significa um classificador perfeito.
    '''


#############################################################################################################################
# Gera um relatório de classificação estendido com micro avg, macro avg, weighted avg, ROC AUC, Average Precision e acurácia.
#############################################################################################################################
def fnc_extended_classification_report(
    y_true,  # Array-like, rótulos verdadeiros
    y_pred,  # Array-like, rótulos preditos
    y_pred_proba,  # Array-like, probabilidades preditas para cada classe
    target_names=None,  # Lista de nomes das classes (opcional)
    flg_mostrar_tabela=True,  # Se True, mostra a tabela de métricas
    flg_retornar_dataframe=False,  # Se True, retorna o dataframe
    precisao=3,  # Número de casas decimais para arredondar a precisão
    paleta_de_cores=PALETTE_TEMPERATURA,  # Paleta de cores para o dataframe
):
    '''
    Gera um relatório de classificação estendido que inclui métricas por classe,
    médias (micro, macro, weighted), ROC AUC por classe e por média,
    Average Precision (Precision-Recall AUC) e acurácia global.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos verdadeiros (Ground truth).
    y_pred : array-like
        Rótulos preditos pelo classificador.
    y_pred_proba : array-like
        Probabilidades preditas para cada classe. Deve ter o formato (n_samples, n_classes).
    target_names : list, optional
        Lista de strings com os nomes das classes. Se None, usa os rótulos numéricos.
    flg_mostrar_tabela : bool, default=True
        Se True, exibe a tabela formatada e colorida (requer `IPython.display.display`).
    flg_retornar_dataframe : bool, default=False
        Se True, retorna o DataFrame pandas contendo as métricas.
    precisao : int, default=3
        Número de casas decimais para arredondamento das métricas ao exibir a tabela.
    paleta_de_cores : str, default=PALETTE_TEMPERATURA
        Paleta de cores (colormap) a ser usada para o destaque condicional na tabela.

    Retorna
    -------
    pandas.DataFrame ou None
        O DataFrame de métricas se `flg_retornar_dataframe` for True,
        caso contrário, retorna None.
    '''

    # Verifica se alguma saída é solicitada; caso contrário, retorna None
    if not (flg_mostrar_tabela or flg_retornar_dataframe):
        return None

    # Obtém o relatório de classificação padrão do sklearn (Precision, Recall, F1, Support)
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    # Inicializa listas para armazenar todas as métricas
    classes = []
    precision = []
    recall = []
    f1 = []
    support = []
    roc_auc = []
    avg_precision = []
    accuracy = []  # A acurácia é a mesma para todas as linhas

    # Determina o número de classes a partir das probabilidades preditas
    n_classes = y_pred_proba.shape[1]

    # Mapeia os nomes das classes para seus índices numéricos (0, 1, 2, ...)
    # Isso é necessário porque o `classification_report` usa strings/nomes
    # e as métricas de AUC/AP usam os índices.
    if target_names is not None:
        label_to_idx = {label: idx for idx, label in enumerate(target_names)}
        idx_to_label = {idx: label for idx, label in enumerate(target_names)}
    else:
        # Se target_names não for fornecido, assume que as classes são 0, 1, 2, ...
        label_to_idx = {str(i): i for i in range(n_classes)}
        idx_to_label = {i: str(i) for i in range(n_classes)}

    # Calcula a acurácia global (único valor para todas as linhas)
    acc = accuracy_score(y_true, y_pred)

    # Dicionário para armazenar métricas de classes individuais (usado para calcular weighted avg)
    class_metrics = {}

    # --- 1. Cálculo das Métricas para Classes Individuais ---
    for cls in report_dict.keys():
        # Ignora as chaves de médias/acurácia
        if cls not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            # Encontra o índice numérico da classe atual
            cls_idx = label_to_idx[cls]

            # Binariza os rótulos verdadeiros: 1 para a classe atual, 0 para todas as outras
            y_true_bin = np.array(
                [1 if y == cls_idx else 0 for y in y_true], dtype=int)

            # Armazena as métricas padrões
            classes.append(cls)
            precision.append(report_dict[cls]['precision'])
            recall.append(report_dict[cls]['recall'])
            f1.append(report_dict[cls]['f1-score'])
            support.append(report_dict[cls]['support'])
            accuracy.append(acc)  # Adiciona a acurácia global

            # Calcula ROC AUC e Average Precision (Precision-Recall AUC) para a classe
            roc_auc_val, ap_val = fnc_compute_roc_auc_and_avg_precision(
                # Usa as probabilidades da coluna da classe
                y_true_bin, y_pred_proba[:, cls_idx])
            roc_auc.append(roc_auc_val)
            avg_precision.append(ap_val)

            # Armazena métricas para cálculo ponderado posterior
            class_metrics[cls] = {
                'roc_auc': roc_auc_val,
                'avg_precision': ap_val,
                'support': report_dict[cls]['support']
            }

    # --- 2. Cálculo das Métricas 'micro avg' ---
    # O `classification_report` não fornece ROC/AP micro, então calculamos F1/Prec/Recall
    # e depois ROC/AP separadamente.

    # Calcula precision, recall, f1 e support para micro avg
    precision_micro, recall_micro, f1_micro, support_micro = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    # Garante que o suporte é o tamanho total da amostra
    if support_micro is None or np.isnan(support_micro):
        support_micro = len(y_true)

    # Armazena as métricas
    classes.append('micro avg')
    precision.append(precision_micro)
    recall.append(recall_micro)
    f1.append(f1_micro)
    support.append(support_micro)
    accuracy.append(acc)  # Adiciona a acurácia global

    # Calcula ROC AUC e Average Precision 'micro avg'
    # Binariza os rótulos verdadeiros para o cálculo multi-classe (necessário para 'micro')
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_true_bin.shape[1] == 1:
        # Lida com o caso binário onde 'label_binarize' retorna apenas 1 coluna
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Calcula ROC AUC e AP com average='micro'
    roc_auc_val, ap_val = fnc_compute_roc_auc_and_avg_precision(
        y_true_bin, y_pred_proba, average='micro')
    roc_auc.append(roc_auc_val)
    avg_precision.append(ap_val)

    # --- 3. Cálculo das Métricas 'macro avg' ---
    classes.append('macro avg')
    # Usa os valores de Precision/Recall/F1 já calculados pelo classification_report
    precision.append(report_dict['macro avg']['precision'])
    recall.append(report_dict['macro avg']['recall'])
    f1.append(report_dict['macro avg']['f1-score'])
    # O suporte é a soma dos suportes das classes individuais
    support.append(sum([
        report_dict[cls]['support']
        for cls in report_dict.keys()
        if cls not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']
    ]))
    accuracy.append(acc)  # Adiciona a acurácia global

    # Calcula ROC AUC e AP 'macro avg' (média simples dos valores por classe)
    valid_metrics = [m for m in class_metrics.values()
                     if not np.isnan(m['roc_auc'])]  # Filtra NaNs
    if valid_metrics:
        roc_auc.append(np.mean([m['roc_auc'] for m in valid_metrics]))
        avg_precision.append(
            np.mean([m['avg_precision'] for m in valid_metrics]))
    else:
        roc_auc.append(np.nan)
        avg_precision.append(np.nan)

    # --- 4. Cálculo das Métricas 'weighted avg' ---
    classes.append('weighted avg')
    # Usa os valores de Precision/Recall/F1 já calculados pelo classification_report
    precision.append(report_dict['weighted avg']['precision'])
    recall.append(report_dict['weighted avg']['recall'])
    f1.append(report_dict['weighted avg']['f1-score'])
    # O suporte é a soma dos suportes das classes individuais
    support.append(sum([
        report_dict[cls]['support']
        for cls in report_dict.keys()
        if cls not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']
    ]))
    accuracy.append(acc)  # Adiciona a acurácia global

    # Calcula ROC AUC e AP 'weighted avg' (média ponderada pelo suporte)
    if valid_metrics:
        weights = np.array([m['support'] for m in valid_metrics])
        weights = weights / np.sum(weights)  # Normaliza os pesos
        roc_auc.append(np.average([m['roc_auc']
                       for m in valid_metrics], weights=weights))
        avg_precision.append(np.average(
            [m['avg_precision'] for m in valid_metrics], weights=weights))
    else:
        roc_auc.append(np.nan)
        avg_precision.append(np.nan)

    # --- 5. Criação e Formatação do DataFrame ---
    df = pd.DataFrame({
        'Classe': classes,
        'Precisão': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Precision-Recall': avg_precision,
        'Acurácia': accuracy,
        'Suporte': support,
    })

    # Otimiza o uso de memória do DataFrame convertendo tipos
    for coluna in df.select_dtypes(include='number').columns:
        # Converte para o menor tipo numérico possível
        for tipo_base in ('float', 'signed', 'unsigned'):
            df[coluna] = pd.to_numeric(df[coluna], downcast=tipo_base)

    # Exibe o DataFrame estilizado
    if flg_mostrar_tabela:
        # A função 'display' deve estar disponível (ex: em ambiente Jupyter/IPython)
        try:
            display(
                df.style.background_gradient(
                    cmap=paleta_de_cores,
                    axis=None,
                    # Aplica o gradiente apenas às colunas de métricas de 0 a 1
                    subset=['Precisão', 'Recall', 'F1-Score',
                            'ROC AUC', 'Precision-Recall', 'Acurácia'],
                    vmin=0.0, vmax=1.0,
                    # Formata a precisão das casas decimais
                ).format(precision=precisao)
            )
        except NameError:
            print(
                "AVISO: 'display' não está disponível. O DataFrame não será exibido com estilo.")
            print(df.to_markdown(index=False, floatfmt=f".{precisao}f"))

    # Retorna o DataFrame, se solicitado
    if flg_retornar_dataframe:
        return df
    else:
        return None


'''
    A precisão indica quantos dos exemplos classificados como positivos pelo 
        modelo realmente pertencem àquela classe. Em outras palavras, é a 
        proporção de verdadeiros positivos sobre o total de exemplos classificados 
        como positivos (verdadeiros positivos + falsos positivos).
    O recall mostra a proporção de verdadeiros positivos sobre o total de 
        exemplos reais daquela classe (verdadeiros positivos + falsos negativos). 
        Ou seja, é a capacidade do modelo de identificar corretamente todos os 
        exemplos de uma determinada classe.
    O f1-score é a média harmônica entre a precisão e o recall. 
        Ele é útil quando há um desbalanceamento entre as classes, pois equilibra 
        os erros da precisão e do recall. Quanto mais próximo de 1, melhor.
    A curva ROC AUC (Receiver Operating Characteristic Area Under the Curve) é 
        uma representação gráfica que relaciona a taxa de verdadeiros positivos 
        e a taxa de falsos positivos em diferentes limiares de classificação. 
        Ela é útil para avaliar a capacidade do modelo de distinguir entre 
        classes positivas e negativas.
    A curva Precision-Recall é uma representação gráfica que relaciona a 
        precisão e o recall em diferentes limiares de classificação. Ela é útil 
        para avaliar o desempenho do modelo em diferentes cenários.
    O suporte indica o número total de exemplos em cada classe.

    Além das métricas por classe, o classification_report oferece médias globais:
        - Accuracy: A acurácia é a proporção de previsões corretas.
        - Micro avg: A média micro é a média ponderada pelas classes,
            considerando o número total de verdadeiros positivos, falsos positivos, 
            verdadeiros negativos e falsos negativos em todas as classes.        
        - Macro avg: A média macro é a média simples das métricas (precision, recall
            e f1-score) para todas as classes, sem considerar o peso de cada classe.
        - Weighted avg: A média ponderada leva em conta o número de exemplos em cada
            classe ao calcular as métricas. Isso é importante quando há desbalanceamento.
'''
