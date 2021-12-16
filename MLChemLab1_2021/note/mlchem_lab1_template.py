#!/usr/bin/env python
#coding=utf-8

"""
Author:        Fanchong Jian, Pengbo Song
Created Date:  2021/09/28
Last Modified: 2021/09/29
"""

from warnings import warn
from sklearn import linear_model

class MLChemLab1(object):
    """Template class for Lab1 -*- Linear Regression -*-

    Properties:
        model: Linear model to fit.
        featurization_mode: Keyword to choose feature methods.
    """

    def __init__(self):
        """Initialize class with empty model and default featurization mode to identical"""
        self.model = None
        self.featurization_mode = "identical"
    
    def fit(self, x, y, featurization_mode : str):
        """Feature input X using given mode and fit model with featurized X and y
        
        Args:
            x: Input X data.
            y: Input y data.
            featurization_mode: Keyword to choose feature methods.

        Returns:
            Trained model using given X and y, or None if no model is specified
        """

        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        self.featurization_mode = featurization_mode
        featurized_x = self.featurization(x)
        self.model.fit(featurized_x, y)
        return self.model
    
    def add_model(self, model : str, **kwargs):
        """Add model before fitting and prediction"""
        if model == "ridge":
            # Put your Ridge Regression model HERE
            self.model = ...
        elif model == "lasso":
            # Put your Lasso Regression model HERE
            self.model = ...
        elif model == "naive":
            # Put your Linear Regression model HERE
            self.model = ...
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + model)

    def featurization(self, x):
        """Feature input X data using preset mode"""
        if self.featurization_mode == "poly":
            # Put your polynomial featurization code HERE
            return ...
        elif self.featurization_mode == "poly-cos":
            # Put your polynomial cosine featurization code HERE
            return ...
        elif self.featurization_mode == "identical":
            # Do nothing, returns raw X data
            return x

    def predict(self, x):
        """Predict based on fitted model and given X data"""
        x = self.featurization(x)
        y_predict = self.model.predict(x)
        return y_predict
    
    def evaluation(self, y_predict, y_label, metric : str):
        """Eavluate training results based on predicted y and true y"""
        if metric == "RMS":
            return ...
        else:
            ...