import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

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

# Initialize the XGBoost Regressor
model = XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'mean Squared Error: {mse}')