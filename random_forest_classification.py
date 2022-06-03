#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:52:23 2022

@author: guilherme
"""

import pandas as pd
from helper import load_normalized_data_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_variables, y_variables = load_normalized_data_classification()

X_train, X_test, y_train, y_test = train_test_split(X_variables, y_variables, test_size=0.3, random_state=10)

clf = RandomForestClassifier(n_estimators=1000, random_state=0)
clf.fit(X_train, y_train)

y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

#previsao
predict = clf.predict(X_test)
df_predict = pd.DataFrame()
df_predict['real'] = y_test
df_predict['previsao'] = predict
df_predict['erro'] = abs(df_predict['real'] - df_predict['previsao'])

fig = plt.figure()
ax = Axes3D(fig)

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
df_plot = X_variables.join(y_variables)
colors = ['red','yellow','orange','blue']
for i in range(1,5):
    df_aux = df_plot[df_plot['classe'] == i]
    ax.scatter(df_aux['qPa'], df_aux['pulso'], df_aux['resp'], color=colors[i-1])


ax.set_xlabel('qPa')
ax.set_ylabel('Pulso')
ax.set_zlabel('Respiração')

plt.show()