import math as M
import numpy as NP
from matplotlib import pyplot as Plot
from sklearn.preprocessing import PolynomialFeatures as PolyFeat
from sklearn.linear_model import RidgeCV as RCV
from sklearn.linear_model import Ridge as R
from sklearn.pipeline import Pipeline as PL
from sklearn.metrics import mean_squared_error as MSE


"""Global Variable Definition"""
TrainData_X = NP.empty([0, 1], float)
TrainData_Y = NP.empty([0, 1], float)
TestData_X1 = NP.empty([0, 1], float)
TestData_Y1 = NP.empty([0, 1], float)
TestData_X2 = NP.empty([0, 1], float)
TestData_Y2 = NP.empty([0, 1], float)
pipe = PL([("poly", PolyFeat()), ("ridge_reg", R())])
pipeCV = PL([("poly", PolyFeat()), ("ridgeCV_reg", RCV())])
FolderPath = "C:\\Workbench\\Computer Science\\Python Programming\\Machine Learning And Chemistry\\Machine Learning And Chemistry\\TaskData1\\"


"""Reading Training Data"""
def ReadSourceData():
    global FolderPath
    SourceName = FolderPath + "dB1-2.dat"
    Source = open(SourceName, mode = "r")
    global TrainData_X, TrainData_Y
    Line = Source.readline()
    while Line:
        TempList = Line.split(",")
        TrainData_X = NP.row_stack((TrainData_X, float(TempList[0])))
        TrainData_Y = NP.row_stack((TrainData_Y, float(TempList[1])))
        Line = Source.readline()
    Source.close()
    QuickSort(TrainData_X, TrainData_Y, 0, TrainData_X.size - 1)


"""Reading Testing Data"""
def ReadTestData():
    global FolderPath
    TestName = FolderPath + "dB1-2-t1.dat"
    Test = open(TestName, mode = "r")
    global TestData_X1, TestData_Y1
    Line = Test.readline()
    while Line:
        TempList = Line.split(",")
        TestData_X1 = NP.row_stack((TestData_X1, float(TempList[0])))
        TestData_Y1 = NP.row_stack((TestData_Y1, float(TempList[1])))
        Line = Test.readline()
    Test.close()
    TestName = FolderPath + "dB1-2-t2.dat"
    Test = open(TestName, mode = "r")
    global TestData_X2, TestData_Y2
    Line = Test.readline()
    while Line:
        TempList = Line.split(",")
        TestData_X2 = NP.row_stack((TestData_X2, float(TempList[0])))
        TestData_Y2 = NP.row_stack((TestData_Y2, float(TempList[1])))
        Line = Test.readline()
    Test.close()
    QuickSort(TestData_X1, TestData_Y1, 0, TestData_X1.size - 1)
    QuickSort(TestData_X2, TestData_Y2, 0, TestData_X2.size - 1)


"""QuickSort for Training and Testing Data"""
def QuickSort(X, Y, Left, Right):
    Mid = (Left + Right) // 2
    i = Left
    j = Right
    while i < j:
        while X[i][0] < X[Mid][0]: i+=1
        while X[j][0] > X[Mid][0]: j-=1
        if i <= j:
            Swap(X, Y, i, j)
            i+=1
            j-=1
    if Left < j: QuickSort(X, Y, Left, j) 
    if i < Right: QuickSort(X, Y, i, Right)
    return X


def Swap(X, Y, i, j):
    tmp = X[i][0]
    X[i][0] = X[j][0]
    X[j][0] = tmp
    tmp = Y[i][0]
    Y[i][0] = Y[j][0]
    Y[j][0] = tmp

"""Model Fitting Check"""
def FittingCheck():
    global pipeCV, FolderPath, TrainData_X, TrainData_Y, TestData_X1, TestData_Y1, TestData_X2, TestData_Y2
    for power in range(1, 11):
        pipeCV.set_params(poly__degree = power)
        pipeCV.set_params(ridgeCV_reg__alphas = NP.logspace(-14, 2, 5000))
        pipeCV.fit(TrainData_X, TrainData_Y)
        TrainPrediction_Y = pipeCV.predict(TrainData_X)
        Train_RMS_score = NP.sqrt(MSE(TrainData_Y, TrainPrediction_Y))
        Plot.xlabel("x")
        Plot.ylabel("y")
        Plot.scatter(TrainData_X, TrainData_Y, label = "Training Data")
        Plot.plot(TrainData_X, TrainPrediction_Y, label = "Predicted Curve")
        Plot.legend()
        Plot.title("Predicted Curve\nα = " + str(pipeCV[1].alpha_) + ", Training RMS = " + str(Train_RMS_score))
        Plot.savefig(FolderPath + "PolyPower_" + str(power) + "_Training_Opt.png", dpi = 300, format = "png")
        Plot.cla()
        TestPrediction_Y1 = pipeCV.predict(TestData_X1)
        Test_RMS_score_1 = NP.sqrt(MSE(TestData_Y1, TestPrediction_Y1))
        Plot.scatter(TestData_X1, TestData_Y1, label = "Test1 Data")
        Plot.plot(TestData_X1, TestPrediction_Y1, label = "Predicted Curve")
        Plot.title("Predicted Curve\nα = " + str(pipeCV[1].alpha_) + ", Test1 RMS = " + str(Test_RMS_score_1))
        Plot.xlabel("x")
        Plot.ylabel("y")
        Plot.legend()
        Plot.savefig(FolderPath + "PolyPower_" + str(power) + "_Test1_Opt.png", dpi = 300, format = "png")
        Plot.cla()
        TestPrediction_Y2 = pipeCV.predict(TestData_X2)
        Test_RMS_score_2 = NP.sqrt(MSE(TestData_Y2, TestPrediction_Y2))
        Plot.scatter(TestData_X2, TestData_Y2, label = "Test2 Data")
        Plot.plot(TestData_X2, TestPrediction_Y2, label = "Predicted Curve")
        Plot.title("Predicted Curve\nα = " + str(pipeCV[1].alpha_) + ", Test2 RMS = " + str(Test_RMS_score_2))
        Plot.xlabel("x")
        Plot.ylabel("y")
        Plot.legend()
        Plot.savefig(FolderPath + "PolyPower_" + str(power) + "_Test2_Opt.png", dpi = 300, format = "png")
        Plot.cla()


"""Model Fitting and Analysis"""
def FittingAndAnalysis():
    global pipe, FolderPath, TrainData_X, TrainData_Y, TestData_X1, TestData_Y1, TestData_X2, TestData_Y2
    OutputName1 = FolderPath + "dLR1a.dat"
    Output1 = open(OutputName1, mode = "w")
    OutputName2 = FolderPath + "dLR1b.dat"
    Output2 = open(OutputName2, mode = "w")
    for power in range(1, 11):
        LogAlpha = NP.empty([0, 1], float)
        Train = NP.empty([0, 1], float)
        Test1 = NP.empty([0, 1], float)
        Test2 = NP.empty([0, 1], float)
        for alpha in NP.logspace(-14, 2, 5000):
            LogAlpha = NP.row_stack((LogAlpha, M.log(alpha)))
            pipe.set_params(poly__degree = power)
            pipe.set_params(ridge_reg__alpha = alpha)
            pipe.fit(TrainData_X, TrainData_Y)
            TrainPrediction_Y = pipe.predict(TrainData_X)
            Train_RMS_score = NP.sqrt(MSE(TrainData_Y, TrainPrediction_Y))
            Train = NP.row_stack((Train, Train_RMS_score))
            TestPrediction_Y1 = pipe.predict(TestData_X1)
            Test_RMS_score_1 = NP.sqrt(MSE(TestData_Y1, TestPrediction_Y1))
            Test1 = NP.row_stack((Test1, Test_RMS_score_1))
            TestPrediction_Y2 = pipe.predict(TestData_X2)
            Test_RMS_score_2 = NP.sqrt(MSE(TestData_Y2, TestPrediction_Y2))
            Test2 = NP.row_stack((Test2, Test_RMS_score_2))
            Output1.write("power = " + str(power) + " alpha = " + str(alpha) +
                          " Train_RMS = " + str(Train_RMS_score) + " Test_RMS_1 = " + str(Test_RMS_score_1) + "\n")
            Output2.write("power = " + str(power) + " alpha = " + str(alpha) +
                          " Train_RMS = " + str(Train_RMS_score) + " Test_RMS_2 = " + str(Test_RMS_score_2) + "\n")
        Plot.plot(LogAlpha, Train, label = "Train")
        Plot.plot(LogAlpha, Test1, label = "Test1")
        Plot.plot(LogAlpha, Test2, label = "Test2")
        Plot.xlabel("ln α")
        Plot.ylabel("RMS")
        Plot.title("Relationship between RMS and ln α")
        Plot.legend()
        Plot.savefig(FolderPath + "PolyPower_" + str(power) + "_RMS.png", dpi = 300, format = "png")
        Plot.cla()
    Output1.close()
    Output2.close()


"""Main Program"""
def main():
    ReadSourceData()
    ReadTestData()
    FittingCheck()
    FittingAndAnalysis()


if __name__ == '__main__':
    main()