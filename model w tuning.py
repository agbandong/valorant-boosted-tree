import pandas as pd
import pickle
from xgboost import XGBRegressor
#from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_excel('VCT_DATASET.xlsx')
test_data = pd.read_excel('Test_VCT_DATASET.xlsx').dropna(how='any',axis=0)

# Select features and target variable
features = ['rounds', 'rating', 'kills_per_round', 'assists_per_round',
            'first_kills_per_round', 'first_deaths_per_round', 'headshot_percentage',
            'clutch_success_percentage', 'total_kills', 'total_deaths', 'total_assists',
            'total_first_kills', 'total_first_deaths']

target_variable = 'average_combat_score'  # Replace with the actual target variable if different

X_train = data[features]
X_test = test_data[features]
y_train = data[target_variable]
y_test = test_data[target_variable]

#Setup Model Pipeline
from sklearn.pipeline import Pipeline

estimators = [
    ('clf', XGBRegressor(random_state=8)) # can customize objective function with the objective parameter
]
pipe = Pipeline(steps=estimators)
pipe

#Set up hyperparameter tuning
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8) 
# in reality, you may consider setting cv and n_iter to higher values

#Train opt
opt.fit(X_train, y_train)
#Evaluate the model and make predictions
print(opt.best_estimator_)
print(opt.best_score_)
y_pred = opt.score(X_test.to_numpy(), y_test.to_numpy())

opt.best_estimator_.steps
from xgboost import plot_importance

xgboost_step = opt.best_estimator_.steps[0]
xgboost_model = xgboost_step[1]
plot_importance(xgboost_model)

print(xgboost_model)

#Save model
filename = 'model w tuning.sav'
pickle.dump(xgboost_model, open(filename, 'wb'))