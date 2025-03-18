#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM Classifier with MinMax Normalization for SPY Data

This script loads a CSV dataset ("SPYV3.csv"), selects a subset of features, and applies 
MinMax normalization to the data. It then performs hyperparameter tuning using 
RandomizedSearchCV for an SVM classifier (with a polynomial kernel), trains the final model 
with the chosen parameters, evaluates it using classification reports and confusion matrices,
and prints the results.

Author: Pablo Beret
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, make_scorer, classification_report
from scipy.stats import uniform as sp_uniform
import warnings

warnings.filterwarnings('ignore')

# --------------------------
# Data Loading and Normalization
# --------------------------
# Carga de datos (se seleccionan las columnas indicadas)
data = pd.read_csv("SPYV3.csv", sep=',', usecols=['1','42','45','47','60',
                                                   '73','171','179','187','221','FECHA.month'])
# Carga de la columna 'CLASIFICADOR'
clasificador = pd.read_csv("SPYV3.csv", sep=',', usecols=['CLASIFICADOR'])

# Normalización de los datos (excluyendo la columna del clasificador)
min_max_scaler = preprocessing.MinMaxScaler()
data_norm = min_max_scaler.fit_transform(data)
data_norm = pd.DataFrame(data_norm, columns=['1','42','45','47','60','73',
                                             '171','179','187','221','FECHA.month'])

# Añadir la columna clasificador
data_norm['CLASIFICADOR'] = clasificador

# Reordenar columnas para que 'CLASIFICADOR' quede primero
data_norm = data_norm[['CLASIFICADOR','1','42','45','47','60','73',
                       '171','179','187','221','FECHA.month']]
del clasificador

# --------------------------
# División del conjunto
# --------------------------
p_train = 0.75  # Porcentaje para entrenamiento
train = data_norm[:int(len(data_norm) * p_train)]
test = data_norm[int(len(data_norm) * p_train):]

print("Training examples:", len(train))
print("Testing examples:", len(test))
print("\n")

# Definición de variables (features y target)
features = data_norm.columns[1:]
x_train = train[features]
y_train = train['CLASIFICADOR']
x_test = test[features]
y_test = test['CLASIFICADOR']

# --------------------------
# Búsqueda de hiperparámetros con RandomizedSearchCV
# --------------------------
clf = SVC()  # Construcción del clasificador SVM

# Construcción de la métrica basada en precision_score
metrica = make_scorer(precision_score, pos_label=1, greater_is_better=True, average="binary")

def report(results, n_top=1):
    """
    Función para mostrar los mejores modelos obtenidos en la búsqueda de hiperparámetros.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {}".format(i))
            print("Mean validation score: {:.3f} (std: {:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {}".format(results['params'][candidate]))
            print("")

# Definición de parámetros y distribuciones a muestrear
param_dist = {
    'class_weight': ['balanced'],
    'C': sp_uniform(1, 40),
    'tol': sp_uniform(0, 1),
    'coef0': sp_uniform(0, 1),
    'random_state': [96],
    'kernel': ['poly'],
    'degree': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'gamma': ['auto'],
    'shrinking': [False]
}

n_iter_search = 4196
random_search = RandomizedSearchCV(clf, scoring=metrica, 
                                   param_distributions=param_dist, 
                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)
report(random_search.cv_results_)

# --------------------------
# Entrenamiento del modelo final con los parámetros obtenidos
# --------------------------
clf_svc = SVC(
    class_weight='balanced',
    C=56.478830222246756, 
    tol=0.4515250769620369, 
    random_state=96,
    kernel='poly',
    degree=11,
    gamma='auto',
    shrinking=False,
    coef0=0.7440671673797636
)
clf_svc.fit(x_train, y_train)
preds = clf_svc.predict(x_test)

# --------------------------
# Evaluación del modelo
# --------------------------
print("SVM Classification Report:\n" + classification_report(y_true=test['CLASIFICADOR'], y_pred=preds))
print("Confusion Matrix:\n")
confusion = pd.crosstab(test['CLASIFICADOR'], preds, rownames=['Actual'], colnames=['Predicted'])
print(confusion)
