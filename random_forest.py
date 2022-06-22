import pandas as pd

from helper import load_normalized_data_regression
from math import sqrt
from sklearn.model_selection import train_test_split

X_variables, y_variables = load_normalized_data_regression()

X_train, X_test, y_train, y_test = train_test_split(X_variables, y_variables, test_size=0.3, random_state=10)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = 0
rmse = 0
mae = 0
for i in range(5):
    rf = RandomForestRegressor(n_estimators = 1000, random_state = i)
    rf.fit(X_train, y_train)
    
    y_test = y_test.reset_index(drop=True)
    
    #previsao
    predict = rf.predict(X_test)

    #df_predict['erro'] = sqrt(pow(df_predict['real'] - df_predict['previsao'], 2))
    
    #resultados
    
    
    mse += mean_squared_error(y_test, predict)
    rmse += sqrt(mse)
    mae += mean_absolute_error(y_test, predict)
    r_squared = r2_score(y_test, predict)

mse = mse / 5
rmse = rmse / 5
mae = mae / 5

df_predict = pd.DataFrame()
df_predict['real'] = y_test
df_predict['previsao'] = predict