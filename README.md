# valorant-boosted-tree
The code for the prediction ACS Valorant model. 
Made by Adrian Bandong for the group thesis of Oratrice.


## Parts: 
- player features prediction model -> The player features prediction model predicts the possible features of a player at a given time. 
- player ACS model -> This is then fed to the player ACS model to return the ACS based on the features of the player.

## Models:
- model no tune new.sav -> ACS model to calculate ACS based on player features using default XGBoost Regressor parameters.
- model new.sav -> ACS model to calculate ACS based on player features with optimized XGBoost Regressor parameters.
- player_feature_models.sav -> Decision Regression Tree model to calculate future player features.
- player_feature_models_linear_limited.sav -> Linear Regression model with maximum and minimum values to calculate future player features.

## Other Models:
- model.sav -> ACS model to calculate ACS based on player features. (Depreciated)
- models new.sav -> All the models that Bayesian Search did to fine-tune model.
- player_feature_models_beta.sav -> Beta regression model to calculate future player features. (Not working) 
- player_feature_models_linear.sav -> Linear Regression model to calculate future player features.
- player_feature_models_XGB.sav -> Defaukt XGBoost Regression models to calculate future player features.

## Python Notebook files:
- ACS_Model.ipynb -> For the training of the main ACS model.
- ACS_Model_test.ipynb -> For the testing of the main ACS model.
- ACS_Prediction_Model.ipynb -> For the training of the feature prediction models.
- ACS_Prediction_Model_test.ipynb -> For the testing of the feature prediction models and ACS prediction model.

## Python files:
- CustomModels.py -> For the customized models that the project needed (Linear with max/min and Beta Regression). 
- model.py / model w tuning.py -> For the training of the main ACS model. (Depreciated)
- test model.py - > For the testing of the main ACS model(Depreciated)

## Excel files:
- SignedPlayersDatasetVALORANT.xlsx -> Main Professional Valorant Player Data
- Signed Players Dataset VALORANT VCT DATASET.xlsx/VCT_DATASET.xlsx/
