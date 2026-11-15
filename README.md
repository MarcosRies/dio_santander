# Detecção de Fraudes em Cartões de Crédito

## O problema

O objetivo do projeto é identificar transações fraudulentas com cartões de crédito.

A base possui muito mais transações normais do que fraudulentas. Por isso, a acurácia sozinha não é suficiente para avaliar o modelo. O mais importante é verificar quantas fraudes ele consegue encontrar.

## Preparação dos dados

Carreguei a base com Pandas e verifiquei a quantidade de fraudes.

Também criei uma variável com o valor das transações em escala logarítmica e utilizei o StandardScaler para padronizar os valores.

Depois, dividi os dados em treino e teste, usando `stratify` para manter a proporção das classes.


## Limiar e SHAP

Também testei o limiar de decisão de 0,3, em vez do padrão 0,5, para tentar identificar mais fraudes.

Utilizei o SHAP para entender quais variáveis tiveram maior influência nas decisões do Random Forest.

As variáveis mais importantes foram: [inserir resultados do SHAP].

## O que foi acrescentado

Usei o código da Expert como ponto de partida e acrescentei:

- Random Forest e XGBoost para comparar os modelos.
- Avaliação com a curva Precision-Recall.
- Teste de outro limiar de decisão.
- GridSearchCV para ajustar os parâmetros.
- Importância das variáveis e análise com SHAP.

Com isso, foi possível comparar diferentes modelos e entender melhor como eles identificam as fraudes.