#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:46:40 2022

@author: guilherme
"""
import pandas as pd

def load_normalized_data_regression():
    X_variables, y_variables = load_data()

    X_variables['qPa'] = (X_variables['qPa'] + 10.0) / 20.0
    X_variables['pulso'] = (X_variables['pulso']) / 200.0
    X_variables['resp'] = (X_variables['resp']) / 22.0
    
    return X_variables, y_variables

def load_normalized_data_classification():
    data = pd.read_csv('sinaisvitais_hist.txt', names=['id', 'pSis', 'pdiast', 'qPa', 'pulso', 'resp', 'grav', 'classe'])

    X_variables = data.iloc[:, 3:6]
    X_variables['qPa'] = (X_variables['qPa'] + 10.0) / 20.0
    X_variables['pulso'] = (X_variables['pulso']) / 200.0
    X_variables['resp'] = (X_variables['resp']) / 22.0
    
    y_variables = data.iloc[:, 7]
    
    return X_variables, y_variables

def load_data():
    data = pd.read_csv('sinaisvitais_hist.txt', names=['id', 'pSis', 'pdiast', 'qPa', 'pulso', 'resp', 'grav', 'classe'])

    X_variables = data.iloc[:, 3:6]
    y_variables = data.iloc[:, 6]
    return X_variables, y_variables

def export_to_json(data):
    import json
    json.dump(data)