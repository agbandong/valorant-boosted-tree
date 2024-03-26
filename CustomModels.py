# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 01:07:19 2024

@author: adban
"""
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

class BetaRegression:
    def __init__(self, df, feature):
        self.feature = feature
        self.feature_min = df[feature].min() * 0.9
        self.feature_max = df[feature].max() * 1.1
        self.model = None
    def __str__(self):
        if self.model is None:
            return None
        return self.model
    def fit(self, X, y):
        y_transformed = (y - self.feature_min) / (self.feature_max - self.feature_min)
        self.model = sm.OLS(y_transformed, sm.add_constant(X)).fit()
        return self
    def predict(self, X):
        if self.model is None:
            return None
        new_value = sm.add_constant(X)
        prediction_transformed = self.model.predict(new_value)
        prediction_original = prediction_transformed * (self.feature_max - self.feature_min) + self.feature_min
        return prediction_original
class LinearLimitedRegression:
    def __init__(self, df, feature):
        self.feature = feature
        self.feature_min = df[feature].min() * 0.9
        self.feature_max = df[feature].max() * 1.1
        self.model = None
    def __str__(self):
        if self.model is None:
            return None
        return self.model
    def fit(self, X, y):
        self.model = LinearRegression().fit(X, y)
    def predict(self, X):
        predictions = self.model.predict(X)
        predictions[predictions < self.feature_min] = self.feature_min
        predictions[predictions > self.feature_max] = self.feature_max
        return(predictions)
#Work in progress
class LinearwXGBoost:
    def __init__(self, linearLimitedModel, XGBoostModel, timeStart, timeEnd):
        self.linearLimitedModel = linearLimitedModel
        self.XGBoostModel = XGBoostModel
        self.timeStart = timeStart
        self.timeEnd = timeEnd