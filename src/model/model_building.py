import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier

from pathlib import Path
PARAMS_PATH = Path("params.yaml")
with PARAMS_PATH.open("r") as f:
    params = yaml.safe_load(f)
n_estimators   = params["model_building"]["n_estimators"]
learning_rate  = params["model_building"]["learning_rate"]
print(n_estimators, learning_rate)
# fetch the data from data/processed
train_data = pd.read_csv('./data/interim/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the XGBoost model

clf = GradientBoostingClassifier(n_estimators=n_estimators , learning_rate=learning_rate)
clf.fit(X_train, y_train)

# save
pickle.dump(clf, open('model.pkl','wb'))