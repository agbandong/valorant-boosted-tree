# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:19:29 2024

@author: adban
"""
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
#Load data
test_data = pd.read_excel('Test Dataset VALORANT VCT DATASET.xlsx').dropna(how='any',axis=0)
features = ['kills_per_round', 'assists_per_round',
            'first_kills_per_round', 'first_deaths_per_round', 'headshot_percentage',
            'clutch_success_percentage', 'total_kills', 'total_deaths', 'total_assists',
            'total_first_kills', 'total_first_deaths']
target_variable = 'average_combat_score'

X_test = test_data[features]
y_test = test_data[target_variable]

#Load Model
filename1 = 'model.sav'
filename2 = 'model w tuning.sav'
model = pickle.load(open(filename1, 'rb'))
opt = pickle.load(open(filename2, 'rb'))

#Test model
y_pred1 = model.predict(X_test.to_numpy())
y_pred2 = opt.predict(X_test.to_numpy())
mse1 = mean_squared_error(y_test, y_pred1)
mse2 = mean_squared_error(y_test, y_pred2)

print(f'Mean Squared Error Default: {mse1}')
print(f'Mean Squared Error with tuning: {mse2}')

plot_importance(opt)