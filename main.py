
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    recall_score
)

from xgboost import XGBClassifier


# Carregando os dados

df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")

print(df.head())


# Proporção de fraudes

fraudes = df["Class"].value_counts(normalize=True)

print(fraudes)


# Feature engineering

df["Amount_log"] = np.log1p(df["Amount"])

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)


# Logistic Regression

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))


# Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)

print("\nRandom Forest:")
print(classification_report(y_test, rf_pred))


# XGBoost

peso = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=peso,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(x_train, y_train)

xgb_pred = xgb.predict(x_test)

print("\nXGBoost:")
print(classification_report(y_test, xgb_pred))


# Comparando os modelos

modelos = {
    "Logistic Regression": model,
    "Random Forest": rf,
    "XGBoost": xgb
}

for nome, modelo in modelos.items():

    y_probs = modelo.predict_proba(x_test)[:, 1]

    print(nome, "- AUC:", roc_auc_score(y_test, y_probs))


# Curva ROC

plt.figure(figsize=(8, 5))

for nome, modelo in modelos.items():

    y_probs = modelo.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.plot(fpr, tpr, label=nome)

plt.title("Comparação das curvas ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# Curva Precision-Recall

plt.figure(figsize=(8, 5))

for nome, modelo in modelos.items():

    y_probs = modelo.predict_proba(x_test)[:, 1]

    precisao, recall, _ = precision_recall_curve(y_test, y_probs)

    plt.plot(recall, precisao, label=nome)

plt.title("Curva Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


# Pipeline e GridSearchCV

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

parametros = {
    "model__C": [0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    parametros,
    scoring="recall",
    cv=3
)

grid.fit(x_train, y_train)

print("\nMelhores parâmetros:")
print(grid.best_params_)


# Testando limiar de 0.3

modelo_final = grid.best_estimator_

y_probs = modelo_final.predict_proba(x_test)[:, 1]

y_pred_03 = (y_probs >= 0.3).astype(int)

print("\nResultados com limiar 0.3:")
print(classification_report(y_test, y_pred_03))


# Importância das variáveis

importancias = pd.Series(
    rf.feature_importances_,
    index=x.columns
)

importancias = importancias.sort_values(ascending=False)

print("\nVariáveis mais importantes:")
print(importancias.head(15))

importancias.head(15).sort_values().plot(kind="barh")

plt.title("Importância das variáveis")
plt.show()


# SHAP

explainer = shap.TreeExplainer(rf)

shap_values = explainer(x_test.iloc[:100])

shap.plots.beeswarm(shap_values[:, :, 1])


# Explicando uma previsão

probabilidades = rf.predict_proba(x_test)[:, 1]

indice = np.argmax(probabilidades)

shap_local = explainer(x_test.iloc[[indice]])

shap.plots.waterfall(shap_local[0, :, 1])

