# Dicionário de Dados - Dataset IBM HR Analytics Employee Attrition & Performance

O dataset, criado por cientistas de dados da IBM, contém informações sobre funcionários de uma empresa fictícia, como idade, gênero, estado civil, nível de satisfação, entre outros. O objetivo é prever a probabilidade de um funcionário sair da empresa. O dataset contém 1.470 observações e 35 variáveis.

[Link do Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")

Os termos *attrition* ("atrito") e *turnover* ("rotatividade") são frequentemente usados de forma intercambiável no contexto de recursos humanos e gestão de força de trabalho, mas eles se referem a conceitos ligeiramente diferentes em relação às mudanças no número e na composição dos empregados dentro de uma organização. Compreender a distinção entre esses dois é importante para analisar e relatar com precisão a dinâmica da força de trabalho.

### Atrito

Atrito refere-se à redução gradual de empregados por meio de circunstâncias naturais, como aposentadoria, demissão voluntária por motivos pessoais, ou a decisão de não substituir empregados que estão de saída. Atrito pode levar a uma diminuição no tamanho da força de trabalho, mas é caracterizado por sua natureza voluntária e muitas vezes incontrolável. Organizações podem permitir o atrito para reduzir o tamanho de sua força de trabalho sem recorrer a demissões, vendo isso como uma forma menos disruptiva de ajustar seus níveis de pessoal.

Em análises, quando você está analisando um conjunto de dados para atrito, você frequentemente olha para as razões por trás das saídas voluntárias e a taxa na qual estas ocorrem naturalmente ao longo do tempo. Taxas de atrito podem fornecer insights sobre a satisfação dos empregados e tendências de longo prazo da força de trabalho.

### Rotatividade

Rotatividade, por outro lado, engloba um escopo mais amplo de partidas de empregados, incluindo tanto renúncias voluntárias quanto demissões involuntárias (como demissões ou términos por justa causa). Porém, ocorre a substituição da força de trabalho. Rotatividade é uma medida de quantos empregados estão deixando a companhia por qualquer motivo. Uma alta taxa de rotatividade pode indicar problemas com satisfação no trabalho, cultura do local de trabalho, ou estabilidade organizacional, e frequentemente requer ação da gestão para endereçar as causas subjacentes.

Rotatividade é calculada dividindo o número de empregados que saem pelo número médio de empregados, geralmente em uma base anual, e expressando isso como uma porcentagem. Esta métrica é crucial para entender como as saídas de empregados, tanto voluntárias quanto involuntárias, afetam a capacidade da organização de manter uma força de trabalho estável e eficaz.

### Diferenças Chave

- **Natureza**: Atrito geralmente é voluntário e pode ser devido a fatores naturais ou incontroláveis, enquanto rotatividade inclui tanto saídas voluntárias quanto demissões involuntárias.
- **Implicações para Gestão**: Atrito pode ser parte de uma estratégia de ajuste natural da força de trabalho e nem sempre é negativo, enquanto alta rotatividade é frequentemente um sinal de problemas dentro da organização que precisam ser abordados.
- **Foco Estratégico**: Análise de atrito pode focar em entender tendências de longo prazo da força de trabalho e planejamento de aposentadoria, enquanto análise de rotatividade é frequentemente usada para identificar questões imediatas com satisfação de empregados, engajamento, ou práticas de gestão que podem estar impactando a retenção de staff.

Enquanto ambos atrito e rotatividade lidam com empregados deixando a organização, eles diferem em seu escopo e as implicações estratégicas para gerenciar e entender a dinâmica da força de trabalho. No seu conjunto de dados de empregados, distinguir entre atrito e rotatividade pode ajudar na customização de estratégias para planejamento da força de trabalho, retenção e desenvolvimento organizacional.

| Coluna                   | Descrição                                                                                                                                                                                              |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Age                      | Idade do funcionário (valor numérico).                                                                                                                                                                 |
| Attrition                | Se o funcionário deixou a empresa (Yes/No).                                                                                                                                                             |
| BusinessTravel           | Frequência de viagens a negócios (Non-Travel, Travel_Frequently, Travel_Rarely).                                                                                                                       |
| DailyRate                | Taxa diária (taxa por dia para a empresa, valor numérico).                                                                                                                                             |
| Department               | Departamento (Human Resources, Research & Development, Sales).                                                                                                                                           |
| DistanceFromHome         | Distância da casa para o trabalho (em unidades, valor numérico).                                                                                                                                       |
| Education                | Nível de educação (valor numérico de 1 a 5). Onde 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor.                                                                                       |
| EducationField           | Área de formação educacional (Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other).                                                                                           |
| EmployeeCount            | Contagem de funcionários (valor numérico constante, geralmente 1).                                                                                                                                     |
| EmployeeNumber           | Número do funcionário (ID único, valor numérico).                                                                                                                                                    |
| EnvironmentSatisfaction  | Satisfação com o ambiente de trabalho (valor numérico de 1 a 4). Onde 1=Low, 2=Medium, 3=High, 4=Very High.                                                                                          |
| Gender                   | Gênero (Male, Female).                                                                                                                                                                                  |
| HourlyRate               | Taxa horária (taxa por hora para a empresa, valor numérico).                                                                                                                                          |
| JobInvolvement           | Envolvimento no trabalho (valor numérico de 1 a 4). Onde 1=Low, 2=Medium, 3=High, 4=Very High.                                                                                                          |
| JobLevel                 | Nível do cargo (valor numérico de 1 a 5).                                                                                                                                                              |
| JobRole                  | Função no trabalho (Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources). |
| JobSatisfaction          | Satisfação com o trabalho (valor numérico de 1 a 4). Onde 1=Low, 2=Medium, 3=High, 4=Very High.                                                                                                      |
| MaritalStatus            | Estado civil (Married, Single, Divorced).                                                                                                                                                                |
| MonthlyIncome            | Renda mensal (salário mensal do funcionário, valor numérico).                                                                                                                                         |
| MonthlyRate              | Taxa mensal (taxa por mês para a empresa, valor numérico).                                                                                                                                            |
| NumCompaniesWorked       | Número de empresas em que trabalhou anteriormente (valor numérico).                                                                                                                                    |
| Over18                   | Se o funcionário tem mais de 18 anos (Yes).                                                                                                                                                             |
| OverTime                 | Se trabalha horas extras (Yes/No).                                                                                                                                                                       |
| PercentSalaryHike        | Percentual de aumento salarial (valor numérico).                                                                                                                                                        |
| PerformanceRating        | Avaliação de desempenho (valor numérico de 1 a 4). Onde 1=Low, 2=Good, 3=Excellent, 4=Outstanding.                                                                                                   |
| RelationshipSatisfaction | Satisfação com relacionamentos no trabalho (valor numérico de 1 a 4). Onde 1=Low, 2=Medium, 3=High, 4=Very High.                                                                                     |
| StandardHours            | Horas padrão de trabalho (valor numérico constante, geralmente 80).                                                                                                                                    |
| StockOptionLevel         | Nível de opções de ações (valor numérico de 0 a 3).                                                                                                                                                |
| TotalWorkingYears        | Total de anos de experiência profissional (valor numérico).                                                                                                                                            |
| TrainingTimesLastYear    | Número de treinamentos no último ano (valor numérico).                                                                                                                                                |
| WorkLifeBalance          | Equilíbrio entre trabalho e vida pessoal (valor numérico de 1 a 4). Onde 1=Bad, 2=Good, 3=Better, 4=Best.                                                                                             |
| YearsAtCompany           | Anos na empresa atual (valor numérico).                                                                                                                                                                 |
| YearsInCurrentRole       | Anos no cargo atual (valor numérico).                                                                                                                                                                   |
| YearsSinceLastPromotion  | Anos desde a última promoção (valor numérico).                                                                                                                                                       |
| YearsWithCurrManager     | Anos com o gerente atual (valor numérico).                                                                                                                                                              |
