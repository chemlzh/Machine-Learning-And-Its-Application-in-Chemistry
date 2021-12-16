#!/usr/bin/env python
#coding=utf-8

"""
Author:        Fanchong Jian, Pengbo Song
Created Date:  2021/09/28
Last Modified: 2021/09/30
"""

from warnings import warn
from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt

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
    
    def fit(self, x, y, featurization_mode : str = "poly", degree : int = 4):
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
        featurized_x = self.featurization(x, degree=degree)
        self.model.fit(featurized_x, y)
        return self.model
    
    def add_model(self, model : str = "ridge", **kwargs):
        """Add model before fitting and prediction"""
        if model == "ridge":
            # Put your Ridge Regression model HERE
            self.model = linear_model.Ridge(**kwargs)
        elif model == "lasso":
            # Put your Lasso Regression model HERE
            self.model = None # TODO
        elif model == "naive":
            # Put your Linear Regression model HERE
            self.model = None # TODO
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + model)

    def featurization(self, x, degree : int = 4):
        """Feature input X data using preset mode"""
        if self.featurization_mode == "poly":
            # Put your polynomial featurization code HERE
            return np.power(x.reshape(-1, 1), np.linspace(1, degree, degree))
        elif self.featurization_mode == "poly-cos":
            # Put your polynomial cosine featurization code HERE
            return x # TODO
        elif self.featurization_mode == "identical":
            # Do nothing, returns raw X data
            return x

    def predict(self, x):
        """Predict based on fitted model and given X data"""
        x = self.featurization(x)
        y_predict = self.model.predict(x)
        return y_predict
    
    def evaluation(self, y_predict, y_label, metric : str = "RMS"):
        """Eavluate training results based on predicted y and true y"""
        if metric == "RMS":
            return metrics.mean_squared_error(y_label, y_predict, squared=True)
        else:
            raise NotImplementedError("Got incorrect metric keyword " + metric)


def parse_dat(filenm: str):
    """Parse DAT file and pack data into numpy float array"""
    xarr = []; yarr = []
    with open(filenm, 'r') as f:
        for line in f.readlines():
            # Read lines in DAT file and split each line by space
            x, y = line.strip().split(' ')
            # Convert splitted values to float and append to the end of list
            xarr.append(float(x))
            yarr.append(float(y))
    # Convert list to numpy array
    xarr = np.asarray(xarr, dtype=float)
    yarr = np.asarray(yarr, dtype=float)
    assert xarr.size == yarr.size, "Got unequal length of X and y vector"
    # Returns array extracted from DAT file
    return xarr, yarr


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
    X_train, y_train = parse_dat("../data/train.dat") # Read train data
    X_test, y_test = parse_dat("../data/test.dat") # Read test data

    # Step 2: Build model
    my_model = MLChemLab1() # Instantiation of the custom class
    my_model.add_model("ridge", alpha=0.5) # Add a model to fit

    # Step 3 & 4: Training
    my_model.fit(X_train, y_train, featurization_mode="poly", degree=4) # Fit model with the train dataset

    # Step 5: Predict
    y_predict = my_model.predict(X_test) # Make prediction using the trained model

    # Step 6: Model evalution
    rmse = my_model.evaluation(y_predict, y_test, metric = "RMS") # Model evaluation with the test dataset
    print(f"RMSE = {rmse:>.5f}")

    # Plot results
    plt.scatter(X_train, y_train, color="green", alpha=0.5)
    plt.scatter(X_test, y_test, color="red", s=0.1)
    plt.plot(X_test, y_predict, color="red")
    plt.ylim(-20, 20)
    plt.savefig("ridge_poly4.png")

if __name__ == "__main__":
    main()
