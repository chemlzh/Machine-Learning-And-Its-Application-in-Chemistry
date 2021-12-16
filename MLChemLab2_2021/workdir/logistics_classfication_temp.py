"""
Author: Zihan Li
Date Created: 2021/10/20
Last Modified: 2021/10/20
Python Version: Anaconda 2021.05 (Python 3.8)
"""

import os
import sys
from warnings import warn

import matplotlib.pyplot as plt
import numpy as npy
import pandas as pd
from sklearn import metrics
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegressionCV as logi_reg_cv
from sklearn.metrics import adjusted_mutual_info_score as adjusted_mis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as grid_cv
from sklearn.preprocessing import KBinsDiscretizer as kbin_disc
from scipy.stats import pearsonr
from scipy.stats import spearmanr

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
            self.model = logi_reg_cv(**kwargs)
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + kw)

    def featurization(self, X):
        """Feature input X data using preset mode"""
        if self.featurization_mode == "normalization":
            # Put your normalization code HERE
            return X.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis = 0)
        elif self.featurization_mode == "standardization":
            X_ave = npy.mean(X); X_stdvar = npy.std(X, ddof = 1)
            return X.apply(lambda x: (x - x.mean()) / x.std(ddof = 1), axis = 0)
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

def plot_bar_and_pie_graph(df_train, trait: str):
    """
    Generate and plot bar graphs and pie graphs
    """
    trait_cnt = df_train[trait].value_counts().sort_index(ascending = True)
    trait_variety = len(trait_cnt)
    response_cnt = df_train["Response"].value_counts().sort_index(ascending = True)
    response_variety = len(response_cnt)
    if ((trait_variety <= 0) and (response_variety <= 0)): 
        raise ValueError("Got empty data!")
    sample_ratio = npy.zeros([trait_variety, response_variety])
    for i in range(0, response_variety):
        res_temp = response_cnt.index[i]
        for j in range(0, trait_variety):
            if (response_cnt[res_temp] == 0): sample_ratio[j][i] = 0.0
            else:
                trait_temp = trait_cnt.index[j]
                cond = ((df_train["Response"] == res_temp) 
                        & (df_train[trait] == trait_temp))
                sample_ratio[j][i] = cond.sum() / response_cnt[res_temp]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), dpi = 300)
    x_location = npy.arange(0.5, 2.5); y_height = npy.zeros(2)
    for i in range(0, trait_variety): 
        ax1.bar(x_location, sample_ratio[i], 0.6, y_height, label = trait_cnt.index[i])
        y_height += sample_ratio[i]
    ax1.set_title("Bar Chart of " + trait + " Ratio \nfor People with Different Response")
    ax1.set_ylim((0, 1.2))
    ax1.set_ylabel("Percentage")
    ax1.set_xticks(x_location)
    ax1.set_xticklabels(["Refuse Insurance", "Accept Insurance"])
    ax1.legend(fancybox = True, framealpha = 1, loc = "upper center", ncol = 2)

    ax2.pie(sample_ratio[:,0], labels = trait_cnt.index, autopct = "%.2f%%", startangle = 90)
    ax2.set_title("Pie Chart of " + trait + " Ratio \nfor People Refusing Insurance")
    ax2.axis("equal") # Draw pie chart as a circle

    ax3.pie(sample_ratio[:,1], labels = trait_cnt.index, autopct = "%.2f%%", startangle = 90)
    ax3.set_title("Pie Chart of " + trait + " Ratio \nfor People Accepting Insurance")
    ax3.axis("equal") # Draw pie chart as a circle

    fig.tight_layout()
    plt.savefig(".\\sample_" + trait + "_ratio_bar_pie_charts.png", bbox_inches = "tight")
    plt.cla()

def plot_box_plot(df_train, trait: str):
    """
    Generate and plot box plot
    """
    response_cnt = df_train["Response"].value_counts().sort_index(ascending = True)
    response_variety = len(response_cnt)
    if (response_variety <= 0): 
        raise ValueError("Got empty data!")
    df_groups = df_train.groupby("Response")
    res_temp = response_cnt.index[0]
    data = []
    for i in range(0, response_variety):
        res_temp = response_cnt.index[i]
        #print(df_groups.get_group(res_temp)[trait])
        data.append(df_groups.get_group(res_temp)[trait].values)
        
    plt.figure(figsize = (8, 6), dpi = 300)
    plt.boxplot(data, labels = ["Refuse Insurance", "Accept Insurance"])
    plt.title("Boxplot of " + trait)
    plt.xlabel("Responce")
    plt.ylabel(trait)
    plt.savefig(".\\sample_" + trait + "_distribution_boxplot.png")
    plt.cla()

def pre_analyze_dataset(df, df_name: str, filepath: str):
    """
    Pre-analyze given datset and print statistical properties of dataset
    """
    output_file = open("statistical_properties_of_dataset_"
                                    + df_name + ".dat", mode = "w")

    gender_cnt = df["Gender"].value_counts(
                            normalize = True).sort_index(ascending = True)
    if ("Male" in gender_cnt.index):
        output_file.write("The percentage of male is {:.2%}\n".format(gender_cnt["Male"]))
    else: output_file.write("The percentage of male is 0.00%\n")
    if ("Female" in gender_cnt.index):
        output_file.write("The percentage of female is {:.2%}\n".format(gender_cnt["Female"]))
    else: output_file.write("The percentage of female is 0.00%\n")
    output_file.write("#" * 20 + "\n")

    output_file.write("The minimum of age is %.2f\n" % df["Age"].min())
    output_file.write("The maximum of age is %.2f\n" % df["Age"].max())
    output_file.write("The mean of age is %.2f\n" % df["Age"].mean())
    output_file.write("The median of age is %.2f\n" % df["Age"].median())
    output_file.write("#" * 20 + "\n")

    driving_license_cnt = df["Driving_License"].value_counts(
                                        normalize = True).sort_index(ascending = True)
    if (0 in driving_license_cnt):
        output_file.write("The percentage of driving-license non-holder is "
              + "{:.2%}\n".format(driving_license_cnt[0]))
    else: print("The percentage of driving-license non-holder is 0.00%\n")
    if (1 in driving_license_cnt):
        output_file.write("The percentage of driving-license holder is "
              + "{:.2%}\n".format(driving_license_cnt[1]))
    else: output_file.write("The percentage of driving-license holder is 0.00%\n")
    output_file.write("#" * 20 + "\n")

    previously_insured_cnt = df["Previously_Insured"].value_counts(
                                            normalize = True).sort_index(ascending = True)
    if (0 in previously_insured_cnt):
        output_file.write("The percentage of previous insurance holder is "
              + "{:.2%}\n".format(previously_insured_cnt[0]))
    else: output_file.write("The percentage of previous insurance holder is 0.00%\n")
    if (1 in previously_insured_cnt):
        output_file.write("The percentage of previous insurance non-holder is "
              + "{:.2%}\n".format(previously_insured_cnt[1]))
    else: output_file.write("The percentage of previous insurance non-holder is 0.00%\n")
    output_file.write("#" * 20 + "\n")

    vehicle_age_cnt = df["Vehicle_Age"].value_counts(
                                    normalize = True).sort_index(ascending = True)
    if ("< 1 Year" in vehicle_age_cnt):
        output_file.write("The percentage of vehicle whose age is less than 1 year is "
                + "{:.2%}\n".format(vehicle_age_cnt["< 1 Year"]))
    else: output_file.write("The percentage of vehicle whose age is less than 1 year is 0.00%\n")
    if ("1-2 Year" in vehicle_age_cnt):
        output_file.write("The percentage of vehicle whose age is about 1 to 2 years is "
                + "{:.2%}\n".format(vehicle_age_cnt["1-2 Year"]))
    else: output_file.write("The percentage of vehicle whose age is about 1 to 2 years is 0.00%\n")
    if ("> 2 Years" in vehicle_age_cnt):
        output_file.write("The percentage of vehicle whose age is more than 2 years is "
                + "{:.2%}\n".format(vehicle_age_cnt["> 2 Years"]))
    else: output_file.write("The percentage of vehicle whose age is more than 2 years is 0.00%\n")
    output_file.write("#" * 20 + "\n")

    vehicle_damage_cnt = df["Vehicle_Damage"].value_counts(
                                        normalize = True).sort_index(ascending = True)
    if ("Yes" in vehicle_damage_cnt):
        output_file.write("The percentage of vehicle damaged before is "
              + "{:.2%}\n".format(vehicle_damage_cnt["Yes"]))
    else: output_file.write("The percentage of vehicle damaged before is 0.00%\n")
    if ("No" in vehicle_damage_cnt):
        output_file.write("The percentage of vehicle not damaged before is "
              + "{:.2%}\n".format(vehicle_damage_cnt["No"]))
    else: output_file.write("The percentage of vehicle not damaged before is 0.00%\n")
    output_file.write("#" * 20 + "\n")

    output_file.write("The minimum of annual premium is %.2f\n" % df["Annual_Premium"].min())
    output_file.write("The maximum of annual premium is %.2f\n" % df["Annual_Premium"].max())
    output_file.write("The mean of annual premium is %.2f\n" % df["Annual_Premium"].mean())
    output_file.write("The median of annual premium is %.2f\n" % df["Annual_Premium"].median())
    output_file.write("#" * 20 + "\n")

    output_file.close()

def discrete_var_mapping(df, trait: str, rule: dict):
    df[trait] = df[trait].map(rule)

def continuous_var_segmentation(df, trait: str, strategy: str, group_num: int):
    proc = kbin_disc(n_bins = group_num, encode = "ordinal", strategy = strategy)
    temp = df[trait].values.reshape(-1, 1)
    proc.fit(temp); df[trait] = proc.transform(temp).astype(npy.int64)

def plot_corr_mat(df_name, mat, tag, method):
    fig, ax = plt.subplots(figsize = (8, 6), dpi = 300)
    hmap_color = plt.cm.RdBu
    if (npy.amax(mat) < 0.01): hmap_color = plt.cm.Reds
    if (npy.amin(mat) > -0.01): hmap_color = plt.cm.Blues
    hmap = ax.matshow(mat, cmap = hmap_color)    
    fig.colorbar(hmap)
    ax.set_xticks(npy.arange(tag.size)); ax.set_yticks(npy.arange(tag.size))
    ax.set_xticklabels(tag); ax.set_yticklabels(tag)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "left", rotation_mode = "anchor")
    fig.tight_layout()
    plt.savefig(".\\" + df_name + "_" + method + "_matrix.png", bbox_inches = "tight")
    plt.cla()

def pre_extract_features(df_name, df_X, df_y):
    df_temp = pd.concat([df_X, df_y], axis = 1)
    if ("id" in df_temp.columns): df_temp.drop("id", axis = 1, inplace=True)
    # print(df_temp)
    var_cnt = df_temp.shape[1]
    pearson_mat = npy.ones((var_cnt, var_cnt), dtype = float)
    spearman_mat = npy.ones((var_cnt, var_cnt), dtype = float)
    mutual_info_mat = npy.ones((var_cnt, var_cnt), dtype = float)
    for i in range(0, var_cnt):
        for j in range(i, var_cnt):
            # print(df_temp.iloc[:, i]); print(df_temp.iloc[:, j])
            pearson_mat[i][j] = pearsonr(df_temp.iloc[:, i].values, 
                                         df_temp.iloc[:, j].values)[0]
            pearson_mat[j][i] = pearson_mat[i][j]
            spearman_mat[i][j] = spearmanr(df_temp.iloc[:, i].values, 
                                         df_temp.iloc[:, j].values)[0]
            spearman_mat[j][i] = spearman_mat[i][j]
            mutual_info_mat[i][j] = adjusted_mis(df_temp.iloc[:, i].values, 
                                            df_temp.iloc[:, j].values)
            mutual_info_mat[j][i] = mutual_info_mat[i][j]
    # print(pearson_mat); print(spearman_mat); print(mutual_info_mat)
    plot_corr_mat(df_name, pearson_mat, df_temp.columns, "pearson")
    plot_corr_mat(df_name, spearman_mat, df_temp.columns, "spearman")
    plot_corr_mat(df_name, mutual_info_mat, df_temp.columns, "adjust_mutual_information")

"""Main testing program"""
def main():
    """General workflow of machine learning
    1. Prepare dataset; 2. Plot statistical graphs;
    3. Split valid dataset; 4. Pre-analyze training set and validating set;
    5. Pre-extract features; 6. Build model;
    7. Training; 8. Predict; 9. Model evalution
    """

    # Random seed setting
    # npy.random.seed(1919810)

    # File path setting
    os.chdir(sys.path[0])
    if (len(sys.argv) == 3):
        train_valid_path = sys.argv[1]
        test_path = sys.argv[2]
    else: 
        train_valid_path = "..\\data\\train_valid.csv"
        test_path = "..\\data\\test.csv"

    # Step 1: Prepare dataset
    df_train_valid = pd.read_csv(train_valid_path, header=0)
    df_test = pd.read_csv(test_path, header=0)
    
    # Step 2: Plot statistical graph
    plot_bar_and_pie_graph(df_train_valid, "Gender")
    plot_bar_and_pie_graph(df_train_valid, "Vehicle_Age")
    plot_bar_and_pie_graph(df_train_valid, "Previously_Insured")
    plot_bar_and_pie_graph(df_train_valid, "Vehicle_Damage")

    plot_box_plot(df_train_valid, "Age")
    plot_box_plot(df_train_valid, "Annual_Premium")

    # Step 3: Split valid dataset
    X_train, X_valid, y_train, y_valid = train_test_split(df_train_valid.drop("Response", axis = 1), 
                                            df_train_valid["Response"], test_size = 0.25, random_state = 114514)
    # print(df_train_valid); print(X_train)
    # print(pd.concat([X_train, y_train], axis = 1))
    pd.concat([X_train, y_train], axis = 1).to_csv(".\\train.csv", index = False)
    pd.concat([X_valid, y_valid], axis = 1).to_csv(".\\valid.csv", index =False)
    X_test = df_test

    # Step 4: Pre-analyze training set and validating set
    pre_analyze_dataset(X_train, "train", ".\\")
    pre_analyze_dataset(X_valid, "valid", ".\\")

    # Step 5: Pre-extract features
    dict_para_list = [["Gender", {"Female": 0, "Male": 1}],
                      ["Vehicle_Age", {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}], 
                      ["Vehicle_Damage", {"No": 0, "Yes": 1}]]
    seg_para_list = [["Age", "uniform", 8],
                     ["Region_Code", "uniform", 8],
                     ["Annual_Premium", "quantile", 12],
                     ["Policy_Sales_Channel", "uniform", 18],
                     ["Vintage", "uniform", 18]]

    for i in range(0, len(dict_para_list)):
        discrete_var_mapping(X_train, dict_para_list[i][0], dict_para_list[i][1])
    for i in range(0, len(seg_para_list)):
        continuous_var_segmentation(X_train, seg_para_list[i][0], 
                                                            seg_para_list[i][1], seg_para_list[i][2])
    pre_extract_features("train_data", X_train, y_train)

    for i in range(0, len(dict_para_list)):
        discrete_var_mapping(X_valid, dict_para_list[i][0], dict_para_list[i][1])
    for i in range(0, len(seg_para_list)):
        continuous_var_segmentation(X_valid, seg_para_list[i][0], 
                                                            seg_para_list[i][1], seg_para_list[i][2])
    pre_extract_features("valid_data", X_valid, y_valid)

    # Step 6: Build model
    my_model = MLChemLab2() # Instantiation of the custom class
    model_C = npy.logspace(-6, 2, 100)
    my_model.add_model("logistic", penalty = "l2", Cs = model_C, 
                                        cv = 10, class_weight = "balanced") # Add a model to fit

    # Step 7: Training
    my_model.fit(X_train, y_train, featurization_mode="normalization") # Fit model with the train dataset

    # Step 8: Predict
    y_valid_pred = my_model.predict(X_valid) # Make prediction using the trained model

    # Step 9: Model evalution
    acc = my_model.evaluation(y_valid, y_valid_pred, metric="accuracy") # Model evaluation with the test dataset
    precision = my_model.evaluation(y_valid, y_valid_pred, metric="precision")
    recall = my_model.evaluation(y_valid, y_valid_pred, metric="recall")
    f1 = my_model.evaluation(y_valid, y_valid_pred, metric="F1")
    print(f"ACCURACY = {acc:>.4f}")
    print(f"PRECISION = {precision:>.4f}")
    print(f"RECALL = {recall:>.4f}")
    print(f"F1 = {f1:>.4f}")

if __name__ == '__main__':
    main()