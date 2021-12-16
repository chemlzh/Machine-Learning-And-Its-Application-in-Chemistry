"""
Author: Zihan Li
Date Created: 2021/09/29
Last Modified: 2021/09/29
Python Version: Anaconda 2021.05 (Python 3.8)
"""

import matplotlib.pyplot as pplot
import numpy as npy
import os
import sys
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import LassoCV as lasso_cv
from sklearn.linear_model import LinearRegression as linear
from sklearn.linear_model import Ridge as ridge
from sklearn.linear_model import RidgeCV as ridge_cv
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures as poly_feat
from warnings import warn

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
        self.is_cv_used = False
        self.power_poly = 0
        self.power_cosine = 0
    
    def read_data(self, path):
        """Reading data from training and testing data"""
        x = []; y = []
        with open(path, mode = "r") as file:
            dataline = file.readline()
            while dataline:
                temp = dataline.split()
                x.append(float(temp[0]))
                y.append(float(temp[1]))
                dataline = file.readline()
            file.close()
            return npy.asarray(x).reshape(-1, 1), npy.asarray(y).reshape(-1, 1)

    def fit(self, x, y, alpha, featurization_mode : str, cv_times):
        """Feature input X using given mode and fit model with featurized X and y
        
        Args:
            x: Input X data.
            y: Input y data.
            alpha: Input hyperparameter alpha, which is a ndarray.
            featurization_mode: Keyword to choose feature methods.
            cv_times: Input fold number for k-fold cross validation.

        Returns:
            Trained model using given X and y, or None if no model is specified
        """

        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        self.featurization_mode = featurization_mode
        featurized_x = self.featurization(x, self.power_poly, self.power_cosine)
        if self.is_cv_used == True:
            self.model.set_params(alphas = alpha, cv = cv_times, 
                                                scoring = "neg_mean_squared_error")
        else:
            self.model.set_params(alpha = alpha)
        self.model.fit(featurized_x, y)
        return self.model
    
    def add_model(self, model : str, **kwargs):
        """Add model before fitting and prediction"""
        if model == "ridge":
            # Put your Ridge Regression model HERE
            self.model = ridge()
            self.is_cv_used = False
        elif model == "ridge_cv":
            # Put your Ridge Regression model with cross-validation HERE
            # self.model = ridge_(alphas = npy.logspace(-23, 2, 10000))
            self.model = ridge_cv()
            self.is_cv_used = True
        elif model == "lasso":
            # Put your Lasso Regression model HERE
            self.model = lasso()
            self.is_cv_used = False
        elif model == "lasso_cv":
            # Put your Lasso Regression model with cross-validation HERE
            # self.model = lasso(alphas = npy.logspace(-23, 2, 10000))
            self.model = lasso_cv()
            self.is_cv_used = True
        elif model == "naive":
            # Put your Linear Regression model HERE
            self.model = linear()
            self.is_cv_used = False
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + model)

    def featurization(self, x, p_poly: int, p_cos: int):
        """Feature input X data using preset mode"""
        if self.featurization_mode == "poly":
            # Put your polynomial featurization code HERE
            if p_poly == 0:
                trans_poly = npy.ones([x.shape[0], 1], dtype = float)
            else:
                temp_feat_poly = poly_feat(degree = p_poly, include_bias = False)
                trans_poly = temp_feat_poly.fit_transform(x)
            return trans_poly
        elif self.featurization_mode == "poly_cos":
            # Put your polynomial cosine featurization code HERE
            if p_cos == 0:
                trans_cos = npy.ones([x.shape[0], 1], dtype = float)
            else:
                temp_feat_cos = poly_feat(degree = p_cos, include_bias = False)
                trans_cos = temp_feat_cos.fit_transform(npy.cos(x))
            return trans_cos
        elif self.featurization_mode == "mixed_poly_cos":
            # Put your mixed polynomial cosine featurization code HERE
            if (p_poly == 0) and (p_cos == 0):
                return npy.ones([x.shape[0], 1], dtype = float)
            else:
                temp_feat_poly = poly_feat(degree = p_poly, include_bias = False)
                temp_feat_cos = poly_feat(degree = p_cos, include_bias = False)
                trans_poly = temp_feat_poly.fit_transform(x)
                trans_cos = temp_feat_cos.fit_transform(npy.cos(x))
                return npy.hstack((trans_poly, trans_cos))
        elif self.featurization_mode == "identical":
            # Do nothing, returns raw X data
            return x

    def predict(self, x):
        """Predict based on fitted model and given X data"""
        x = self.featurization(x, self.power_poly, self.power_cosine)
        y_predict = self.model.predict(x)
        return y_predict
    
    def evaluation(self, y_predict, y_label, metric : str):
        """Evaluate training results based on predicted y and true y"""
        if metric == "RMS":
            return mse(y_label, y_predict, squared = False)
        else:
            raise NotImplementedError("Got incorrect metric keyword " + metric)
    
    def greater_or_equal(self, a: float, b: float):
        return ((a > b) or npy.isclose(a, b))

    def lesser_or_equal(self, a: float, b: float):
        return ((a < b) or npy.isclose(a, b))

    def rmse_coef_alpha_relation(self, min_pow, max_pow, train_data_x, train_data_y,
                                                test_data_x, test_data_y, mod_type: str, feat_type: str):
        """
        Sub program for task 1 and task 3, used to determine and plot relation 
        between RMS, coefficient and hyperparameter alpha
        """
        output_file = open("manually_optimized_alpha_for_" + feat_type + "_using_" 
                                        + mod_type + "_method.dat", mode = "w")
        self.add_model(mod_type)
        #mlchem.add_model("ridge_cv")
        alpha_cnt = 10000
        alpha_chosen = npy.logspace(-23, 2, alpha_cnt)
        pow_chosen = npy.fromiter(iter(range(min_pow, max_pow + 1)), dtype = int)
        best_alpha = 0.0; best_power = 0; best_power_attached = 0
        best_train_mse = float(0x7fffffff); best_test_mse = float(0x7fffffff)
        best_coef = npy.empty([max_pow + 1], dtype = float) 

        """
        Manual optimization procedure
        """
        for mod_pow in pow_chosen:
            if feat_type == "poly":
                self.power_poly = mod_pow
            elif feat_type == "poly_cos":
                self.power_cosine = mod_pow
            else:
                raise NotImplementedError("Got incorrect feature keyword "
                                                            + feat_type + " for task 1 or task 3")
            mod_train_mse = npy.empty([alpha_cnt, 1], dtype = float)
            mod_test_mse = npy.empty([alpha_cnt, 1], dtype = float)
            mod_coef = npy.empty([alpha_cnt, mod_pow + 1], dtype = float)
            optimized_alpha =0.0
            optimized_train_mse = float(0x7fffffff); optimized_test_mse = float(0x7fffffff)
            optimized_predicted_data_y = npy.empty([0, 1], dtype = float)
            optimized_coef = npy.empty([mod_pow + 1], dtype = float)
            for i in range(0, alpha_cnt):
                self.fit(train_data_x, train_data_y, alpha_chosen[i], feat_type, 0)
                mod_coef[i][0] = self.model.intercept_
                for p in range(0, mod_pow):
                    mod_coef[i][p + 1] = self.model.coef_[0][p]
                predicted_train_data_y = self.predict(train_data_x)
                mod_train_mse[i][0] = self.evaluation(predicted_train_data_y, 
                                                                            train_data_y, "RMS")
                predicted_test_data_y = self.predict(test_data_x)
                mod_test_mse[i][0] = self.evaluation(predicted_test_data_y, 
                                                                            test_data_y, "RMS")
                if (self.lesser_or_equal(mod_train_mse[i][0], optimized_train_mse)
                    and (mod_test_mse[i][0] < optimized_test_mse)):
                    optimized_alpha = alpha_chosen[i]
                    optimized_train_mse = mod_train_mse[i][0]
                    optimized_test_mse = mod_test_mse[i][0]
                    optimized_predicted_train_data_y = predicted_train_data_y.copy()
                    optimized_predicted_test_data_y = predicted_test_data_y.copy()
                    optimized_coef = mod_coef[i].copy()

            """
            Output of manual optimization result
            """
            if feat_type == "poly":
                output_file.write("Manually optimized alpha for power_poly = " 
                                            + str(mod_pow) + ": " 
                                            + str(optimized_alpha) + "\n")
            elif feat_type == "poly_cos":
                output_file.write("Manually optimized alpha for power_cosine = " 
                                            + str(mod_pow) + ": " 
                                            + str(optimized_alpha) + "\n")
            output_file.write("Training RMSE = " + str(optimized_train_mse) + "\n")
            output_file.write("Testing RMSE = " + str(optimized_test_mse) + "\n")
            output_file.write("C[0] = " + str(optimized_coef[0]) + "\n")
            for p in range(1, mod_pow + 1): 
                output_file.write("C[" + str(p) + "] = " + str(optimized_coef[p]) + "\n")
            output_file.write("#" * 20 + "\n")
            if optimized_test_mse < best_test_mse:
                best_alpha = optimized_alpha
                best_power = mod_pow
                best_train_mse = optimized_train_mse
                best_test_mse = optimized_test_mse
                best_coef = optimized_coef

            """
            Plot of manually optimized fitting model for different mod_pow
            """
            pplot.scatter(train_data_x, train_data_y, 
                                label = "Training Data", color = "tab:orange")
            pplot.scatter(test_data_x, test_data_y, 
                                label = "Testing Data", color = "tab:blue")
            pplot.plot(test_data_x, optimized_predicted_test_data_y, 
                                label = "Fitting Model", color = "tab:blue")
            pplot.xlabel("x"); pplot.ylabel("y")
            pplot.title("Training Data, Testing Data and Optimized Fitting Model\n"
                            + "α = " + str(optimized_alpha) 
                            + " , Testing RMSE = " + str(optimized_test_mse))
            pplot.legend()
            pplot.savefig("power_" + feat_type + "_" + str(mod_pow) 
                                + "_fitting_model.png", dpi = 300, format = "png")
            pplot.cla()

            """
            Plot of relation between RMS, coefficient and hyperparameter alpha
            """
            pplot.plot(npy.log10(alpha_chosen), mod_train_mse, 
                            label = "Training RMSE")
            pplot.plot(npy.log10(alpha_chosen), mod_test_mse,
                          label = "Testing RMSE")
            pplot.xlabel("Log10 λ"); pplot.ylabel("RMSE")
            pplot.title("Relationship Between RMSE and Log10 λ")
            pplot.legend()
            pplot.savefig("power_" + feat_type + "_" + str(mod_pow) 
                                + "_rmse.png", dpi = 300, format = "png")
            pplot.cla()
            for p in range(0, mod_pow + 1):
                pplot.plot(npy.log10(alpha_chosen), mod_coef[:, p], 
                             label = "C[" + str(p) + "]")
            pplot.xlabel("Log10 λ"); pplot.ylabel("Coefficient")
            pplot.title("Relationship Between Coefficient and Log10 λ")
            pplot.legend()
            pplot.savefig("power_" + feat_type + "_" + str(mod_pow) 
                                + "_coefficient.png", dpi = 300, format = "png")
            pplot.cla()
        
        """
        Output of best fitting model, using an alternative solution of task 2 and 4
        """
        output_file.write("These are parameters of the best "
                                    + feat_type +" model: \n")
        output_file.write("power_poly = " + str(best_power) + "\n")
        output_file.write("power_cosine = " + str(best_power_attached) + "\n")
        output_file.write("Optimized alpha = " + str(best_alpha) + "\n")
        output_file.write("Traning RMSE = " + str(best_train_mse) + "\n")
        output_file.write("Testing RMSE = " + str(best_test_mse) + "\n")
        output_file.write("C[0] = " + str(best_coef[0]) + "\n")
        for p in range(1, best_power + 1): 
            output_file.write("C[" + str(p) + "]_poly = " 
                                        + str(best_coef[p]) + "\n")
        for p in range(best_power + 1, best_power + best_power_attached + 1):
            output_file.write("C[" + str(p - best_power) + "]_poly_cos = " 
                                        + str(best_coef[p]) + "\n")
        output_file.close()

    def find_optimized_alpha(self, min_pow, max_pow, min_pow_attached, 
                                    max_pow_attached, train_data_x, train_data_y,
                                    test_data_x, test_data_y, mod_type: str, feat_type: str, cv_times):
        """
        Sub program for task 2, 4 and 6, used to find optimized alpha for 
        different model with a variety of power, and to give the best poly 
        model, poly_cos model and mixed_poly_cos model

        Note that allowed min value of power for mixed_poly_cos model 
        is 1, though self.fit() can fit model with power_poly = 0 or power
        _cosine = 0, since when power_poly or power_cosine becomes 
        zero, the model will degenerate into poly or poly_cos model, and
        these model can be done with feat_type = "poly" or "poly_cos"
        """
        output_file = open("auto_optimized_alpha_for_" + feat_type + "_using_" 
                                        + mod_type + "_method.dat", mode = "w")
        self.add_model(mod_type)
        if self.is_cv_used == False:
            raise NotImplementedError("Got incorrect model keyword " + mod_type 
                                + " for task 2, 4 and 6, since these tasks need cross validation")
        alpha_cnt = 10000
        alpha_chosen = npy.logspace(-23, 2, alpha_cnt)
        best_alpha = 0.0; best_power = 0; best_power_attached = 0
        best_train_mse = float(0x7fffffff); best_test_mse = float(0x7fffffff)
        output_file.write("This model is verified by " + str(cv_times) 
                                    + "-fold" + "cross validation" + "\n")
        output_file.write("#" * 20 + "\n")
        if feat_type == "mixed_poly_cos":     
            best_coef = npy.empty([max_pow + max_pow_attached + 1], dtype = float)
            pow_chosen = npy.fromiter(
                iter(range(min_pow, max_pow + 1)), dtype = int)
            pow_attached_chosen = npy.fromiter(
                iter(range(min_pow_attached, max_pow_attached + 1)), dtype = int)

            """
            Auto optimization procedure for mixed_poly_cos using cross validation
            """
            for mod_pow in pow_chosen:
                for mod_pow_attached in pow_attached_chosen:
                    self.power_poly = mod_pow
                    self.power_cosine = mod_pow_attached
                    mod_coef = npy.empty([mod_pow + mod_pow_attached + 1], dtype = float)
                    self.fit(npy.vstack((train_data_x, test_data_x)), 
                             npy.vstack((train_data_y, test_data_y)), 
                             alpha_chosen, feat_type, cv_times)
                    mod_coef[0] = self.model.intercept_
                    for p in range(0, mod_pow + mod_pow_attached):
                        mod_coef[p + 1] = self.model.coef_[0][p]
                    predicted_train_data_y = self.predict(train_data_x)
                    mod_train_mse = self.evaluation(predicted_train_data_y, 
                                                                        train_data_y, "RMS")
                    predicted_test_data_y = self.predict(test_data_x)
                    mod_test_mse = self.evaluation(predicted_test_data_y, 
                                                                        test_data_y, "RMS")

                    """
                    Output of auto optimization result for mixed_poly_cos
                    """
                    output_file.write("Auto optimized alpha for power_poly = " 
                                      + str(mod_pow) + " and power_cosine = " 
                                      + str(mod_pow_attached) + ": " 
                                      + str(self.model.alpha_) + "\n")
                    output_file.write("Training RMSE = " + str(mod_train_mse) + "\n")
                    output_file.write("Testing RMSE = " + str(mod_test_mse) + "\n")
                    output_file.write("C[0] = " + str(mod_coef[0]) + "\n")
                    for p in range(1, mod_pow + 1): 
                        output_file.write("C[" + str(p) + "]_poly = " 
                                                    + str(mod_coef[p]) + "\n")
                    for p in range(mod_pow + 1, mod_pow + mod_pow_attached + 1):
                        output_file.write("C[" + str(p - mod_pow) + "]_poly_cos = " 
                                                    + str(mod_coef[p]) + "\n")
                    output_file.write("#" * 20 + "\n")
                    if mod_test_mse < best_test_mse:
                        best_alpha = self.model.alpha_
                        best_power = mod_pow
                        best_power_attached = mod_pow_attached
                        best_train_mse = mod_train_mse
                        best_test_mse = mod_test_mse
                        best_coef = mod_coef.copy()
                                
                    """
                    Plot of auto optimized mixed_poly_cos fitting model 
                    for different mod_pow and mod_pow_attached
                    """
                    pplot.scatter(train_data_x, train_data_y, 
                                        label = "Original training Data", color = "tab:orange")
                    pplot.scatter(test_data_x, test_data_y, 
                                        label = "Original testing Data", color = "tab:blue")
                    pplot.plot(test_data_x, predicted_test_data_y, 
                                    label = "Fitting Model", color = "tab:blue")
                    pplot.xlabel("x"); pplot.ylabel("y")
                    pplot.title("Training Data, Testing Data and Optimized Fitting Model\n"
                                    + "α = " + str(self.model.alpha_) 
                                    + " , Testing RMSE = " + str(mod_test_mse))
                    pplot.legend()
                    pplot.savefig("power_poly_" + str(mod_pow) 
                                        + "_power_poly_cos_" + str(mod_pow_attached)
                                        + "_fitting_model_using_cv.png", dpi = 300, format = "png")
                    pplot.cla()

            """
            Output of best mixed_poly_cos fitting model
            """
            output_file.write("These are parameters of the best "
                              + feat_type +" model: \n")
            output_file.write("power_poly = " + str(best_power) + "\n")
            output_file.write("power_cosine = " + str(best_power_attached) + "\n")
            output_file.write("Optimized alpha = " + str(best_alpha) + "\n")
            output_file.write("Traning RMSE = " + str(best_train_mse) + "\n")
            output_file.write("Testing RMSE = " + str(best_test_mse) + "\n")
            output_file.write("C[0] = " + str(best_coef[0]) + "\n")
            for p in range(1, best_power + 1): 
                output_file.write("C[" + str(p) + "]_poly = " 
                                            + str(best_coef[p]) + "\n")
            for p in range(best_power + 1, best_power + best_power_attached + 1):
                output_file.write("C[" + str(p - best_power) + "]_poly_cos = " 
                                            + str(best_coef[p]) + "\n")
            output_file.close()
        else:
            best_coef = npy.empty([max_pow + 1], dtype = float)       
            pow_chosen = npy.fromiter(
                iter(range(min_pow, max_pow + 1)), dtype = int)

            """
            Auto optimization procedure for poly (or poly_cos) using cross validation
            """
            for mod_pow in pow_chosen:
                if feat_type == "poly":
                    self.power_poly = mod_pow
                elif feat_type == "poly_cos":
                    self.power_cosine = mod_pow
                else:
                    raise NotImplementedError("Got incorrect feature keyword "
                                                                + feat_type + " for task 2 and 4")
                mod_coef = npy.empty([mod_pow + 1], dtype = float)
                self.fit(npy.vstack((train_data_x, test_data_x)), 
                        npy.vstack((train_data_y, test_data_y)), 
                        alpha_chosen, feat_type, cv_times)
                mod_coef[0] = self.model.intercept_
                for p in range(0, mod_pow):
                    mod_coef[p + 1] = self.model.coef_[0][p]
                predicted_train_data_y = self.predict(train_data_x)
                mod_train_mse = self.evaluation(predicted_train_data_y, 
                                                                    train_data_y, "RMS")
                predicted_test_data_y = self.predict(test_data_x)
                mod_test_mse = self.evaluation(predicted_test_data_y, 
                                                                    test_data_y, "RMS")

                """
                Output of auto optimization result for poly (or poly_cos)
                """
                if feat_type == "poly":
                    output_file.write("Auto optimized alpha for power_poly = " 
                                            + str(mod_pow) + ": " 
                                            + str(self.model.alpha_) + "\n")
                elif feat_type == "poly_cos":
                    output_file.write("Auto optimized alpha for power_cosine = " 
                                            + str(mod_pow) + ": " 
                                            + str(self.model.alpha_) + "\n")
                output_file.write("Training RMSE = " + str(mod_train_mse) + "\n")
                output_file.write("Testing RMSE = " + str(mod_test_mse) + "\n")
                output_file.write("C[0] = " + str(mod_coef[0]) + "\n")
                for p in range(1, mod_pow + 1): 
                    output_file.write("C[" + str(p) + "] = " + str(mod_coef[p]) + "\n")
                output_file.write("#" * 20 + "\n")
                if mod_test_mse < best_test_mse:
                    best_alpha = self.model.alpha_
                    best_power = mod_pow
                    best_train_mse = mod_train_mse
                    best_test_mse = mod_test_mse
                    best_coef = mod_coef.copy()

                """
                Plot of manually optimized fitting model for different mod_pow
                """
                pplot.scatter(train_data_x, train_data_y, 
                                    label = "Original training Data", color = "tab:orange")
                pplot.scatter(test_data_x, test_data_y, 
                                    label = "Original testing Data", color = "tab:blue")
                pplot.plot(test_data_x, predicted_test_data_y, 
                                label = "Fitting Model", color = "tab:blue")
                pplot.xlabel("x"); pplot.ylabel("y")
                pplot.title("Training Data, Testing Data and Optimized Fitting Model\n"
                                + "α = " + str(self.model.alpha_) 
                                + " , Testing RMSE = " + str(mod_test_mse))
                pplot.legend()
                pplot.savefig("power_" + feat_type + "_" + str(mod_pow) 
                                    + "_fitting_model_using_cv.png", dpi = 300, format = "png")
                pplot.cla()
            
            """
            Output of best poly (or poly_cos) fitting model
            """
            output_file.write("These are parameters of the best "
                              + feat_type +" model: \n")
            if feat_type == "poly":
                output_file.write("power_poly = " + str(best_power) + "\n")
            elif feat_type == "poly_cos":
                output_file.write("power_cosine = " + str(best_power) + "\n")
            output_file.write("Optimized alpha = " + str(best_alpha) + "\n")
            output_file.write("Traning RMSE = " + str(best_train_mse) + "\n")
            output_file.write("Testing RMSE = " + str(best_test_mse) + "\n")
            output_file.write("C[0] = " + str(best_coef[0]) + "\n")
            for p in range(1, best_power + 1): 
                output_file.write("C[" + str(p) + "] = " + str(best_coef[p]) + "\n")
            output_file.close()
    
    def fit_special_model(self, train_data_x, train_data_y,
                                        test_data_x, test_data_y, mod_type: str):
        """
        Sub program for task 5, using manual optimization
        """
        output_file = open("fitting_result_of_special_model_by_manual_optimization" 
                                        + "_using_" + mod_type + "_method.dat", mode = "w")
        self.add_model(mod_type)
        if self.is_cv_used == True:
            raise NotImplementedError("Got incorrect model keyword " + mod_type 
                                + " for task 5_1, since these tasks need manual optimization")
        alpha_cnt = 10000
        alpha_chosen = npy.logspace(-23, 2, alpha_cnt)
        self.power_poly = 0; self.power_cosine = 4
        mod_coef = npy.empty([self.power_poly + self.power_cosine + 1], 
                                            dtype = float)
        optimized_alpha =0.0
        optimized_train_mse = float(0x7fffffff); optimized_test_mse = float(0x7fffffff)
        optimized_predicted_data_y = npy.empty([0, 1], dtype = float)
        optimized_coef = npy.empty([self.power_cosine + 1], dtype = float)
        for i in range(0, alpha_cnt):
            self.fit(train_data_x, train_data_y, alpha_chosen[i], "poly_cos", 0)
            mod_coef[0] = self.model.intercept_
            for p in range(0, self.power_cosine):
                mod_coef[p + 1] = self.model.coef_[0][p]
            predicted_train_data_y = self.predict(train_data_x)
            mod_train_mse = self.evaluation(predicted_train_data_y + train_data_x, 
                                                            train_data_y, "RMS")
            predicted_test_data_y = self.predict(test_data_x)
            mod_test_mse = self.evaluation(predicted_test_data_y + test_data_x, 
                                                            test_data_y, "RMS")
            if (self.lesser_or_equal(mod_train_mse, optimized_train_mse) 
                and (mod_test_mse < optimized_test_mse)):
                    optimized_alpha = alpha_chosen[i]
                    optimized_train_mse = mod_train_mse
                    optimized_test_mse = mod_test_mse
                    optimized_predicted_train_data_y = predicted_train_data_y.copy()
                    optimized_predicted_test_data_y = predicted_test_data_y.copy()
                    optimized_coef = mod_coef.copy()

        """
        Output of optimization result
        """
        output_file.write("Optimized alpha for special model: "  
                                    + str(optimized_alpha) + "\n")
        output_file.write("Training RMSE = " + str(optimized_train_mse) + "\n")
        output_file.write("Testing RMSE = " + str(optimized_test_mse) + "\n")
        output_file.write("C[0] = " + str(optimized_coef[0]) + "\n")
        for p in range(1, self.power_poly + self.power_cosine + 1): 
            output_file.write("C[" + str(p) + "]_poly_cos = " + str(optimized_coef[p]) + "\n")
        output_file.close()

        """
        Plot of manually optimized special model
        """
        pplot.scatter(train_data_x, train_data_y, 
                                label = "Training Data", color = "tab:orange")
        pplot.scatter(test_data_x, test_data_y, 
                                label = "Testing Data", color = "tab:blue")
        pplot.plot(test_data_x, optimized_predicted_test_data_y + test_data_x, 
                                label = "Fitting Model", color = "tab:blue")
        pplot.xlabel("x"); pplot.ylabel("y")
        pplot.title("Training Data, Testing Data and Optimized Fitting Model\n"
                            + "α = " + str(optimized_alpha) 
                            + " , Testing RMSE = " + str(optimized_test_mse))
        pplot.legend()
        pplot.savefig("manually_optimized_special_model.png", dpi = 300, format = "png")
        pplot.cla()

    def fit_special_model_cv(self, train_data_x, train_data_y,
                                        test_data_x, test_data_y, mod_type: str, cv_times):
        """
        Sub program for task 5, using cross validation
        """
        output_file = open("fitting_result_of_special_model_by_auto_optimization" 
                                        + "_using_" + mod_type + "_method.dat", mode = "w")
        self.add_model(mod_type)
        if self.is_cv_used == False:
            raise NotImplementedError("Got incorrect model keyword " + mod_type 
                                + " for task 5_2, since these tasks need cross validation")
        alpha_cnt = 10000
        alpha_chosen = npy.logspace(-23, 2, alpha_cnt)
        self.power_poly = 0; self.power_cosine = 4
        mod_coef = npy.empty([self.power_poly + self.power_cosine + 1], 
                                            dtype = float)
        output_file.write("This model is verified by " + str(cv_times) 
                                    + "-fold " + "cross validation" + "\n")
        self.fit(npy.vstack((train_data_x, test_data_x)), 
                    npy.vstack((train_data_y, test_data_y)), 
                    alpha_chosen, "poly_cos", cv_times)
        mod_coef[0] = self.model.intercept_
        for p in range(0, self.power_poly + self.power_cosine):
            mod_coef[p + 1] = self.model.coef_[0][p]
        predicted_train_data_y = self.predict(train_data_x)
        mod_train_mse = self.evaluation(predicted_train_data_y + train_data_x, 
                                                            train_data_y, "RMS")
        predicted_test_data_y = self.predict(test_data_x)
        mod_test_mse = self.evaluation(predicted_test_data_y + test_data_x, 
                                                            test_data_y, "RMS")
        
        """
        Output of optimization result
        """
        output_file.write("Optimized alpha for special model: "  
                                    + str(self.model.alpha_) + "\n")
        output_file.write("Training RMSE = " + str(mod_train_mse) + "\n")
        output_file.write("Testing RMSE = " + str(mod_test_mse) + "\n")
        output_file.write("C[0] = " + str(mod_coef[0]) + "\n")
        for p in range(1, self.power_poly + self.power_cosine + 1): 
            output_file.write("C[" + str(p) + "]_poly_cos = " + str(mod_coef[p]) + "\n")
        output_file.close()

        """
        Plot of auto optimized special model
        """
        pplot.scatter(train_data_x, train_data_y, 
                                label = "Original training Data", color = "tab:orange")
        pplot.scatter(test_data_x, test_data_y, 
                                label = "Original testing Data", color = "tab:blue")
        pplot.plot(test_data_x, predicted_test_data_y + test_data_x, 
                                label = "Fitting Model", color = "tab:blue")
        pplot.xlabel("x"); pplot.ylabel("y")
        pplot.title("Training Data, Testing Data and Optimized Fitting Model\n"
                            + "α = " + str(self.model.alpha_) 
                            + " , Testing RMSE = " + str(mod_test_mse))
        pplot.legend()
        pplot.savefig("auto_optimized_special_model.png", dpi = 300, format = "png")
        pplot.cla()

"""Main testing program"""
def main():
    npy.random.seed(1919810)

    """Read training and testing data"""
    mlchem = MLChemLab1()
    os.chdir(sys.path[0])
    if (len(sys.argv) == 3):
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    else: 
        train_path = "..\\data\\train.dat"
        test_path = "..\\data\\test.dat"
    train_data_x, train_data_y = mlchem.read_data(train_path)
    test_data_x, test_data_y = mlchem.read_data(test_path)

    """
    Task 1: Fit with polynomial model, then determine and plot relation 
    between RMS, coefficient and hyperparameter alpha
    """
    mlchem.rmse_coef_alpha_relation(0, 4, train_data_x, train_data_y, 
                                                        test_data_x, test_data_y, "ridge", "poly")

    """
    Task 2: Find optimized alpha for polynomial model using cross validation
    """
    mlchem.find_optimized_alpha(0, 4, 0, 4, train_data_x, train_data_y, 
                                                    test_data_x, test_data_y, "ridge_cv", "poly", 5)

    """
    Task 3: Fit with polynomial-cosine model, then determine and plot 
    relation between RMS, coefficient and hyperparameter alpha
    """
    mlchem.rmse_coef_alpha_relation(0, 4, train_data_x, train_data_y, 
                                                        test_data_x, test_data_y, "ridge", "poly_cos")

    """
    Task 4: Find the optimized alpha for polynomial-cosine model 
    using cross validation
    """
    mlchem.find_optimized_alpha(0, 4, 0, 4, train_data_x, train_data_y, 
                                               test_data_x, test_data_y, "ridge_cv", "poly_cos", 5)

    """
    Task 5: Fit with y = C_0 + x + C_1 cos x+ C_2 cos^2 x + C_3 cos^3 x + 
    C_4 cos^4 x, then print optimized coefficient and RMSE
    """
    mlchem.fit_special_model(train_data_x, train_data_y, 
                                            test_data_x, test_data_y, "ridge")

    mlchem.fit_special_model_cv(train_data_x, train_data_y, 
                                            test_data_x, test_data_y, "ridge_cv", 5)

    """
    Task 6: Find the optimized alpha for mixed-polynomial-cosine model 
    using cross validation
    """
    mlchem.find_optimized_alpha(1, 4, 1, 4, train_data_x, train_data_y, 
                                                test_data_x, test_data_y, "ridge_cv", "mixed_poly_cos", 5)

if __name__ == '__main__':
    main()