import pandas as PD
import numpy as NP
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as PL
from sklearn import metrics


"""Global Variable Definition"""
FolderPath = "C:\\Workbench\\Computer Science\\Python Programming\\Machine Learning And Chemistry\\Machine Learning And Chemistry\\TaskData2\\"
pipeCV = PL([("LogRegCV", LRCV(Cs = NP.logspace(-2, 4, 1000), cv = 5, multi_class = 'multinomial'))])
SalesMap = { "accounting":-0.30, "hr": -0.56, "IT": 0.17, "management": 1.00, "marketing": 0.02, "product_mng": 0.20, "RandD": 0.90, "sales": -0.07, "support": -0.12, "technical": -0.19 }
SalaryMap = { "low": -0.34, "medium": 0.20, "high": 1.00 }


"""Reading Data"""
def ReadData():
    global FolderPath, pipeCV, SalaryMap, SalesMap
    global TrainData_X, TrainData_Y, TestData_X, TestData_Y
    source = PD.read_csv(FolderPath + "HR_comma_sep.csv")
    source["salary"] = source["salary"].replace(SalaryMap)
    source["sales"] = source["sales"].replace(SalesMap)
    X = source[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "sales", "salary"]].values
    Y = source["left"].values
##    TrainData_X, TestData_X, TrainData_Y, TestData_Y = train_test_split(X, Y, test_size = 0.8, random_state = 1551)
    TrainData_X, TestData_X, TrainData_Y, TestData_Y = train_test_split(X, Y, test_size = 0.8)
    

"""Model Training"""
def Training():
    global FolderPath, pipeCV
    global TrainData_X, TrainData_Y
    pipeCV.fit(TrainData_X, TrainData_Y)
    TrainPrediction_Y = pipeCV.predict(TrainData_X)
    OutputName1 = FolderPath + "dLogisticR2a.dat"
    Output1 = open(OutputName1, mode = "w")
    Output1.write("satisfaction_level    last_evaluation    number_project    average_montly_hours    time_spend_company    Work_accident    promotion_last_5years    sales    salary    left_prediction    left_result\n")
    for i in range(len(TrainData_X)):
        tmp = ""
        for j in TrainData_X[i]:
            tmp = tmp + str(j) + "    "
        tmp = tmp + str(TrainPrediction_Y[i]) + "    " + str(TrainData_Y[i]) + "\n"
        Output1.write(tmp)
    Output1.write("\n")
    Output1.write("Accuracy score = " + str(metrics.accuracy_score(TrainData_Y, TrainPrediction_Y)) + "\n")
    Output1.write("AUC = " + str(metrics.roc_auc_score(TrainData_Y, TrainPrediction_Y)) + "\n")
    Output1.close()


"""Model Analysing"""
def Analysing():
    global FolderPath, pipeCV
    global TestData_X, TestData_Y
    TestPrediction_Y = pipeCV.predict(TestData_X)
    OutputName2 = FolderPath + "dLogisticR2b.dat"
    Output2 = open(OutputName2, mode = "w")
    Output2.write("satisfaction_level    last_evaluation    number_project    average_montly_hours    time_spend_company    Work_accident    promotion_last_5years    sales    salary    left_prediction    left_result\n")
    for i in range(len(TestData_X)):
        tmp = ""
        for j in TestData_X[i]:
            tmp = tmp + str(j) + "    "
        tmp = tmp + str(TestPrediction_Y[i]) + "    " + str(TestData_Y[i]) + "\n"
        Output2.write(tmp)
    Output2.write("\n")
    Output2.write("Accuracy score = " + str(metrics.accuracy_score(TestData_Y, TestPrediction_Y)) + "\n")
    Output2.write("AUC = " + str(metrics.roc_auc_score(TestData_Y, TestPrediction_Y)) + "\n")
    Output2.close()


"""Printing Coefficient"""
def PrintCoef():
    global FolderPath, pipeCV
    OutputName3 = FolderPath + "dLogisticR2c.dat"
    Output3 = open(OutputName3, mode = "w")
    Output3.write("C = " + str(pipeCV[0].C_[0]) + "\n")
    for i in range(len(pipeCV[0].coef_[0])):
        Output3.write("coef" + str(i + 1) + " = " + str(pipeCV[0].coef_[0][i]) + "\n")
    Output3.write("intercept = " + str(pipeCV[0].intercept_[0]) + "\n")
    Output3.close()
    

"""Main Program"""
def main():
    ReadData()
    Training()
    Analysing()
    PrintCoef()


if __name__ == '__main__':
    main()