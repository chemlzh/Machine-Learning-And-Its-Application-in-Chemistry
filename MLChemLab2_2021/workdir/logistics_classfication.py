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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler as RUS
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV as logi_reg_cv
from sklearn.metrics import adjusted_mutual_info_score as adjusted_mis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as grid_cv
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
    output_file = open(filepath + "statistical_properties_of_dataset_"
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

def continuous_var_segmentation(df, trait: str, group_num: int,
                                            include_pinf: bool, utrunc: float):
    if (include_pinf == True):
        seg = npy.linspace(df[trait].min(), utrunc, group_num)
        seg = npy.append(seg, 0x7fffffff)
    else: seg = npy.linspace(df[trait].min(), utrunc, group_num + 1)
    df[trait] = pd.cut(x = df[trait], bins = seg, include_lowest = True, labels = False)

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
    if ("id" in df_temp.columns): df_temp.drop("id", axis = 1, inplace = True)
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

def auto_prediction_and_output(model: MLChemLab2, X, trait,
                               filename: str = "default_predict", filepath: str = ".\\"):
    pred = model.predict(X[trait]) 
    pd.concat([X["id"], pd.Series(pred, name = "Response")], 
        axis = 1).sort_values(by = ["id"]).to_csv(filepath + filename + ".csv", index = False)
    return pred

def model_evaluation(model: MLChemLab2, y_true, y_pred, eval_indicator: list,
                                model_name: str = "default model", filepath: str = ".\\"):
    output_file = open(filepath + "evaluation_of_"
                                    + model_name + ".dat", mode = "w")
    output_file.write("The hyperparameter C equals to " + 
                      "{:.4e}".format(model.model.C_[0]) + "\n")
    output_file.write("#" * 20 + "\n")
    eval_cnt = len(eval_indicator)
    for i in range(0, eval_cnt): 
        eval_res = model.evaluation(y_true, y_pred, metric = eval_indicator[i])
        output_file.write(eval_indicator[i] + " = {:.4f}".format(eval_res) + "\n")
    output_file.close()


"""Main testing program"""
def main():
    """General workflow of machine learning
    1. Prepare dataset; 2. Plot statistical graphs;
    3. Split valid dataset; 4. Pre-analyze training set and validating set;
    5. Pre-extract features; 6. Over-sampling or under-sampling;
    7. Build model; 8. Training; 
    9. Predict; 10. Model evalution
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
    pd.concat([X_train, y_train], axis = 1).sort_values(
        by = ["id"]).to_csv(".\\train.csv", index = False)
    pd.concat([X_valid, y_valid], axis = 1).sort_values(
        by = ["id"]).to_csv(".\\valid.csv", index = False)
    X_test = df_test

    # Step 4: Pre-analyze training set and validating set
    pre_analyze_dataset(X_train, "train", ".\\")
    pre_analyze_dataset(X_valid, "valid", ".\\")

    # Step 5: Pre-extract features
    dict_para_list = [["Gender", {"Female": 0, "Male": 1}],
                      ["Vehicle_Age", {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 1}], 
                      ["Vehicle_Damage", {"No": 0, "Yes": 1}]]
    seg_para_list = [["Age", 8, False, df_train_valid["Age"].max()],
                     ["Region_Code", 8, False, df_train_valid["Region_Code"].max()],
                     ["Annual_Premium", 9, True, 100000],
                     ["Policy_Sales_Channel", 18, False, df_train_valid["Policy_Sales_Channel"].max()],
                     ["Vintage", 18, False, df_train_valid["Vintage"].max()]]

    for i in range(0, len(dict_para_list)):
        discrete_var_mapping(X_train, dict_para_list[i][0], dict_para_list[i][1])
    for i in range(0, len(seg_para_list)):
        continuous_var_segmentation(X_train, seg_para_list[i][0], seg_para_list[i][1], 
                                                            seg_para_list[i][2], seg_para_list[i][3])
    pre_extract_features("train_data", X_train, y_train)
    #pd.concat([X_train, y_train], axis = 1).to_csv(".\\train_transformed.csv", index = False)

    for i in range(0, len(dict_para_list)):
        discrete_var_mapping(X_valid, dict_para_list[i][0], dict_para_list[i][1])
    for i in range(0, len(seg_para_list)):
        continuous_var_segmentation(X_valid, seg_para_list[i][0], seg_para_list[i][1], 
                                                            seg_para_list[i][2], seg_para_list[i][3])
    pre_extract_features("valid_data", X_valid, y_valid)
    #pd.concat([X_valid, y_valid], axis = 1).to_csv(".\\valid_transformed.csv", index = False)

    for i in range(0, len(dict_para_list)):
        discrete_var_mapping(X_test, dict_para_list[i][0], dict_para_list[i][1])
    for i in range(0, len(seg_para_list)):
        continuous_var_segmentation(X_test, seg_para_list[i][0], seg_para_list[i][1], 
                                                            seg_para_list[i][2], seg_para_list[i][3])

    # Step 6: Over-sampling or under-sampling
    sm = SMOTE(sampling_strategy = "minority", random_state = 1919810)
    # rus = RUS(random_state = 1919810)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    # X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # print(X_train_res.value_counts())
    pre_extract_features("train_data_after_resampling", X_train_res, y_train_res)

    # Step 7: Build model
    my_model = MLChemLab2() # Instantiation of the custom class
    my_model_trait_sele = MLChemLab2()
    model_C = npy.logspace(-5, 5, 101)
    # my_model.add_model("logistic", penalty = "l2", Cs = model_C, 
    #                                     cv = 10, class_weight = "balanced") 
    my_model.add_model("logistic", penalty = "l2", Cs = model_C, 
                            cv = 10, class_weight = "balanced", max_iter = 500) 
    my_model_trait_sele.add_model("logistic", penalty = "l2", Cs = model_C, 
                            cv = 10, class_weight = "balanced", max_iter = 500) 
    # Add a model to fit

    # Step 8: Training
    # my_model.fit(X_train, y_train, featurization_mode = "standardization") 
    my_model.fit(X_train_res.drop("id", axis = 1), y_train_res, featurization_mode = "normalization") 
    trait_sele = ["Previously_Insured", "Vehicle_Age", "Vehicle_Damage"]
    my_model_trait_sele.fit(X_train_res[trait_sele], y_train_res, featurization_mode = "normalization") 
    # Fit model with the train dataset

    # Step 9: Predict
    y_train_pred = auto_prediction_and_output(my_model, 
        X_train, X_train.columns.drop("id"), "train_predict", ".\\")
    y_valid_pred = auto_prediction_and_output(my_model, 
        X_valid, X_valid.columns.drop("id"), "valid_predict", ".\\")
    y_test_pred = auto_prediction_and_output(my_model, 
        X_test, X_test.columns.drop("id"), "test_predict", ".\\")

    y_train_pred_trait_sele = auto_prediction_and_output(my_model_trait_sele, 
                                                    X_train, trait_sele, "train_predict_trait_sele", ".\\")
    y_valid_pred_trait_sele = auto_prediction_and_output(my_model_trait_sele, 
                                                    X_valid, trait_sele, "valid_predict_trait_sele", ".\\")
    y_test_pred_trait_sele = auto_prediction_and_output(my_model_trait_sele, 
                                                    X_test, trait_sele, "test_predict_trait_sele", ".\\")
    # Make prediction using the trained model

    # Step 10: Model evalution
    eval_indicator = ["accuracy", "precision", "recall", "F1", "AUC"]
    model_evaluation(my_model, y_train, y_train_pred, 
                     eval_indicator, "train_logistic_reg_all_traits", ".\\")
    model_evaluation(my_model, y_valid, y_valid_pred, 
                     eval_indicator, "valid_logistic_reg_all_traits", ".\\")

    model_evaluation(my_model_trait_sele, y_train, y_train_pred_trait_sele, 
                     eval_indicator, "train_logistic_reg_selected_traits", ".\\")
    model_evaluation(my_model_trait_sele, y_valid, y_valid_pred_trait_sele, 
                     eval_indicator, "valid_logistic_reg_selected_traits", ".\\")
    # Model evaluation with the test dataset
    
    # print(f"ACCURACY = {acc:>.4f}")
    # print(f"PRECISION = {precision:>.4f}")
    # print(f"RECALL = {recall:>.4f}")
    # print(f"F1 = {f1:>.4f}")
    
if __name__ == '__main__':
    main()