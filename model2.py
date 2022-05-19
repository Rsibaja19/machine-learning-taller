# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


simplefilter(action='ignore', category=FutureWarning)


def me(model, x_train, x_test, y_train, y_test):
    # metricas de entrenamiento
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred


def mca(model, x_test, y_test, y_pred):
    # matriz de confusion auc
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC


def fpr_tpr(model, x_test, y_test):
    # matriz de fpr y tpr
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    return fpr, tpr


def show_roc_hot(matriz_confusion):
    # show hot plot ROC
    for i in range(len(matriz_confusion)):
        sns.heatmap(matriz_confusion[i])
    plt.show()