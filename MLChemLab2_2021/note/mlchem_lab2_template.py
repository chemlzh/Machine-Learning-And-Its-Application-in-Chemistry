#!/usr/bin/env python
#coding=utf-8

"""
Author:        Fanchong Jian, Pengbo Song
Created Date:  2021/10/15
Last Modified: 2021/10/17
"""

from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

class MLChemLab2(object):
    """Template class for Lab2 -*- Logistic Regression -*-

    Properties:
        model: Logistic model to fit.
        featurization_mode: Keyword to choose feature methods.
    """

    def __init__(self):
        """Initialize class with empty model and default featurization mode to identical"""
        self.model = None
        self.featurization_mode = "identical"
    
    def fit(self, X, y, featurization_mode : str = "normalization"):
        """Feature input X using given mode and fit model with featurized X and y
        
        Args:
            X: Input X data.
            y: Input y data.
            featurization_mode: Keyword to choose feature methods.

        Returns:
            Trained model using given X and y, or None if no model is specified
        """
        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        # Set featurization mode keyword
        self.featurization_mode = featurization_mode
        # Preprocess X
        featurized_X = self.featurization(X)
        self.model.fit(featurized_X, y)
        return self.model
    
    def add_model(self, kw : str = "logistic", **kwargs):
        """Add model before fitting and prediction

        Args:
            kw: Keyword that indicates which model to build.
            kwargs: Keyword arguments passed to 
        """
        if kw == "logistic":
            self.model = linear_model.LogisticRegression(**kwargs)
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + kw)

    def featurization(self, X):
        """Feature input X data using preset mode"""
        if self.featurization_mode == "normalization":
            # Put your normalization code HERE
            return X # TODO
        elif self.featurization_mode == "identical":
            # Do nothing, returns raw X data
            return X

    def predict(self, X):
        """Predict based on fitted model and given X data"""
        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        X = self.featurization(X)
        y_pred = self.model.predict(X)
        return y_pred
    
    def evaluation(self, y_true, y_pred, metric : str = "accuracy"):
        """Eavluate training results based on predicted y and true y"""
        if metric == "accuracy":
            return metrics.accuracy_score(y_true, y_pred)
        elif metric == "precision":
            return metrics.precision_score(y_true, y_pred)
        elif metric == "recall":
            return metrics.recall_score(y_true, y_pred)
        elif metric == "F1":
            return metrics.f1_score(y_true, y_pred)
        elif metric == "CM":
            return metrics.confusion_matrix(y_true, y_pred)
        elif metric == "AUC":
            return metrics.roc_auc_score(y_true, y_pred)
        else:
            raise NotImplementedError("Got incorrect metric keyword " + metric)


def main():
    """General workflow of machine learning
    1. Prepare dataset
    2. Build model
    3. Data preprocessing (featurization, normalization, ...)
    4. Training
    5. Predict
    6. Model evalution
    """

    # Step 1: Prepare dataset
    df_train = pd.read_csv("../data/train.csv", header=0)
    df_test = pd.read_csv("../data/test.csv", header=0)
    X_train_valid = df_train[["Driving_License", "Previously_Insured"]]
    y_train_valid = df_train["Response"]
    # Split valid dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25)
    X_test = df_test[["Driving_License", "Previously_Insured"]]

    # Step 2: Build model
    my_model = MLChemLab2() # Instantiation of the custom class
    my_model.add_model("logistic", penalty="l2", C=1.0, class_weight="balanced") # Add a model to fit

    # Step 3: Data preprocessing    
    # Step 4: Training
    my_model.fit(X_train, y_train, featurization_mode="normalization") # Fit model with the train dataset

    # Step 5: Predict
    y_valid_pred = my_model.predict(X_valid) # Make prediction using the trained model

    # Step 6: Model evalution
    acc = my_model.evaluation(y_valid, y_valid_pred, metric="accuracy") # Model evaluation with the test dataset
    precision = my_model.evaluation(y_valid, y_valid_pred, metric="precision")
    recall = my_model.evaluation(y_valid, y_valid_pred, metric="recall")
    f1 = my_model.evaluation(y_valid, y_valid_pred, metric="F1")
    print(f"ACCURACY = {acc:>.4f}")
    print(f"PRECISION = {precision:>.4f}")
    print(f"RECALL = {recall:>.4f}")
    print(f"F1 = {f1:>.4f}")

if __name__ == "__main__":
    main()
