import numpy as np
import pandas as pd
from dvclive import Live
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score ,f1_score
import yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)



clf = pickle.load(open('model.pkl','rb'))
test_data = pd.read_csv('./data/interim/test_bow.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

with Live(save_dvc_exp=True) as live:

    live.log_metric("accuracy", accuracy)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1", f1)


    for param, value in params.items():
        for key, val in value.items():

            live.log_param(f'{param}_{key}', val)
with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)