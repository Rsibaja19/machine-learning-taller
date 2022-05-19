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


def show_roc_curve_matrix(model, x_test, y_test):
    colors = ['orange', 'blue', 'yellow', 'green', 'red', 'silver']
    # show plot ROC
    for i in range(len(model)):
        fpr, tpr = fpr_tpr(model[i], x_test, y_test)
        # sns.heatmap(matriz_confusion)
        # plt.show()
        plt.plot(fpr, tpr, color=colors[i], label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(model_name + ['line'])
    plt.show()


def sh_me(str_model, AUC, acc_validation, acc_test, y_test, y_pred):
    # show metrics
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')


def convert_data_categ(data, columns):
    # convert data to category
    for column in columns:
        rows = list(data[column].unique())
        new_rows = list(range(len(rows)))
        data.replace(rows, new_rows, inplace=True)


def panda_values(data):
    columns = ['Accuracy de Entrenamiento', 'Accuracy de Validación',
               'Accuracy de Test', 'Recall del Modelo', 'Precisión del Modelo',
               'F1-Score del Modelo', 'Área bajo la Curva (AUC)']
    data = np.transpose(data)
    tabla = pd.DataFrame(data=data, index=model_name, columns=columns)
    return tabla.sort_values(by=['Área bajo la Curva (AUC)'], ascending=False)


def view_matriz_confusion(matriz_confusion):
    for i in range(len(matriz_confusion)):
        print(model_name[i])
        print(pd.DataFrame(matriz_confusion[i]))
        print('\n')


url = 'DataSets/Bank Marketing/bank-full.csv'
data = pd.read_csv(url)

# manejo de datos
data.y.value_counts(dropna=False)

columns = ['job', 'marital', 'default', 'loan', 'contact', 'y']

# Volver categoricos los datos
convert_data_categ(data, columns)

# caso unico (education)
rows = ['tertiary', 'secondary', 'primary']
new_rows = [3, 2, 1]
data.education.replace(rows, new_rows, inplace=True)

rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
# contact, day, month, duration, previous
columns_delete = ['contact', 'day', 'month', 'duration',
                  'previous', 'pdays', 'balance', 'poutcome']
data.drop(columns_delete, axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)

# machine learning
x = np.array(data.drop(['y'], 1))
y = np.array(data.y)  # 0 no suscrito, 1 suscrito

# x_train, x_test, y_train, y_test
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)

# metricas
model_name = ['LOGISTIC REGRESSION', 'DECISION TREE', 'KNEIGHBORNS',
              'RANDOM FOREST CLASSIFIER', 'GRADIENT BOOSTING CLASSIFIER']

acc_va = acc_te = recall = precision = f1 = auc = list(range(len(model_name)))
matriz_confu = list(range(len(model_name)))
model = [LogisticRegression(), DecisionTreeClassifier(),
         KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(),
         GradientBoostingClassifier()]

for i in range(len(model_name)):
    # model, acc_validation, acc_test, y_pred
    model[i], vacc_va, vacc_te, y_pr = me(model[i], x_tr, x_te, y_tr, y_te)
    matriz_confusion, vauc = mca(model[i], x_te, y_te, y_pr)
    # valores de la matriz de confusion
    recall[i] = recall_score(y_te, y_pr)
    precision[i] = precision_score(y_te, y_pr)
    f1[i] = f1_score(y_te, y_pr)
    auc[i] = vauc
    acc_va[i] = vacc_va
    acc_te[i] = vacc_te
    # matriz de confusion
    matriz_confu[i] = matriz_confusion

tabla = panda_values([acc_va, acc_va, acc_te, recall, precision, f1, auc])
view_matriz_confusion(matriz_confu)
show_roc_hot(matriz_confu)
print(tabla)
show_roc_curve_matrix(model, x_te, y_te)
