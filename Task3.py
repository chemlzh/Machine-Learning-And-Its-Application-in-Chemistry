from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.pipeline import Pipeline as PL
from sklearn import metrics
import numpy as NP
import joblib as JL


"""Global Variable Definition and Third-party Model Import"""
FolderPath = "C:\\Workbench\\Computer Science\\Python Programming\\Machine Learning And Chemistry\\Machine Learning And Chemistry\\TaskData3\\"
pipe = PL([("MLPClassifier", MLPC(solver="adam", activation="logistic", alpha=1e-4, random_state=1551, 
                                  hidden_layer_sizes=(200, 100, 50, 25, ), max_iter=500, learning_rate_init=1e-3))])
#pipe = PL([("MLPClassifier", MLPC(solver="adam", activation="logistic", alpha=1e-4, random_state=1551, 
#                                  hidden_layer_sizes=(50, ), max_iter=500, learning_rate_init=1e-3))])
import sys
sys.path.append(FolderPath)
import mlc_mnist as mnist


def ReadData():
    global FolderPath
    global TrainImages, TrainLabels, TestImages, TestLabels
    TrainImages, TrainLabels = mnist.load_mnist(FolderPath, "train")
    TestImages, TestLabels = mnist.load_mnist(FolderPath, "t10k")
    TrainImages.shape
    TrainLabels.shape


def ModelTraining():
    global pipe, TrainImages, TrainLabels
    pipe.fit(TrainImages, TrainLabels)


def ModelAssessment():
    global pipe, TestImages, TestLabels
    TestPrediction = pipe.predict(TestImages)
    ModelConfusionMatrix = metrics.confusion_matrix(TestLabels, TestPrediction)
    ModelScore = pipe.score(TestImages, TestLabels)
    ModelMicroPrecision = metrics.precision_score(TestLabels, TestPrediction, average="micro")
    ModelMacroPrecision = metrics.precision_score(TestLabels, TestPrediction, average="macro")
    ModelWeightedPrecision = metrics.precision_score(TestLabels, TestPrediction, average="weighted")
    ModelMicroRecallRate = metrics.recall_score(TestLabels, TestPrediction, average="micro")
    ModelMacroRecallRate = metrics.recall_score(TestLabels, TestPrediction, average="macro")
    ModelWeightedRecallRate = metrics.recall_score(TestLabels, TestPrediction, average="weighted")
    OutputName = FolderPath + "dNeuralNReport3.dat"
    Output = open(OutputName, mode = "w")
    Output.write("Now printing confusion matrix...\n")
    Output.write("T \ P\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n")
    for i in range(0, 10):
        tmp = "  " + str(i) + "  "
        for j in range(0, 10):
            tmp += "\t" + str(ModelConfusionMatrix[i][j])
        Output.write(tmp + "\n")
    Output.write("\n")
    Output.write("Now printing precision rate and recall rate...\n")
    Output.write("Model score = " + str(ModelScore) + "\n")
    Output.write("Micro Precision rate = " + str(ModelMicroPrecision) + "\n")
    Output.write("Macro Precision rate = " + str(ModelMacroPrecision) + "\n")
    Output.write("Weighted Precision rate = " + str(ModelWeightedPrecision) + "\n")
    Output.write("Micro Recall rate = " + str(ModelMicroRecallRate) + "\n")
    Output.write("Macro Recall rate = " + str(ModelMacroRecallRate) + "\n")
    Output.write("Weighted Recall rate = " + str(ModelWeightedRecallRate) + "\n")
    print("Confusion matrix = \n", ModelConfusionMatrix)
    print()
    print("Now printing precision rate and recall rate...")
    print("Model score = ", ModelScore)
    print("Micro Precision rate = ", ModelMicroPrecision)
    print("Macro Precision rate = ", ModelMacroPrecision)
    print("Weighted Precision rate = ", ModelWeightedPrecision)
    print("Micro Recall rate = ", ModelMicroRecallRate)
    print("Macro Recall rate = ", ModelMacroRecallRate)
    print("Weighted Recall rate = ", ModelWeightedRecallRate)
    JL.dump(pipe, FolderPath + "dNeuralN3.dump")


def main():
    ReadData()
    ModelTraining()
    ModelAssessment()


if __name__ == '__main__':
    main()