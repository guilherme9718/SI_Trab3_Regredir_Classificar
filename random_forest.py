import pandas as pd
import numpy as np

data = pd.read_csv('sinaisvitais_hist.txt', names=['id', 'pSis', 'pdiast', 'qPa', 'pulso', 'resp', 'grav', 'classe'])
print(data)


from sklearn.model_selection import train_test_split

X_variables = data.iloc[:, 3:6]
X_variables['qPa'] = (X_variables['qPa'] + 10.0) / 20.0
X_variables['pulso'] = (X_variables['pulso']) / 200.0
X_variables['resp'] = (X_variables['resp']) / 22.0
print(X_variables)
y_variables = data.iloc[:, 6]

X_train, X_test, y_train, y_test = train_test_split(X_variables, y_variables, test_size=0.3)

from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(X_train, y_train)

predict = rf.predict(X_test)
print(predict)