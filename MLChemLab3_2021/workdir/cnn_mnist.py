"""
Author: Zihan Li
Date Created: 2021/11/3
Last Modified: 2021/11/3
Python Version: Anaconda 2021.05 (Python 3.8)
"""

import os
import sys
from warnings import warn

import torch
import torchvision
import pandas as pd
import numpy as npy
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# custom Dataset wrapper for new handwritten digits organization
class mySet(Dataset):
    def __init__(self, images):
        super(mySet, self).__init__()
        self.data = images
    def __getitem__(self, x): return self.data[x]
    def __len__(self): return len(self.data)

class CNN(nn.Module):
    def __init__(self, conv1c: int = 16, conv2c: int = 32, 
            conv1k: int = 5, conv2k: int = 3, fc1: int = 128, fc2: int = 10, 
            batchnorm: bool = True, dropout: float = 0.1): 
        """
        Parameters are not required. If you introduce any parameters 
        for convenience, their default values should be specified.
        """
        super(CNN, self).__init__()
        # put your layers here
        self.conv1 = nn.Conv2d(1, conv1c, kernel_size = (conv1k,conv1k))
        self.conv2 = nn.Conv2d(conv1c, conv2c, kernel_size = (conv2k,conv2k))

        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout(p = dropout)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(conv1c)
            self.bn2 = nn.BatchNorm2d(conv2c)
        
        final_size = ((28 - conv1k + 1) // 2 - conv2k + 1) // 2
        self.fc1 = nn.Linear(conv2c * final_size * final_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        # out = YOUR_LAYERS(x)
        # assume conv1k = 5 and conv2k = 3
        out = self.conv1(x) # [batch_size, conv1c, 24, 24]
        out = F.relu(self.pool(out)) # [batch_size, conv1c, 12, 12]
        if self.batchnorm:
            out = self.bn1(out) # [batch_size, conv1c, 12, 12]

        out = self.conv2(out) # [batch_size, conv2c, 10, 10]
        out = F.relu(self.pool(out)) # [batch_size, conv2c, 5, 5]
        if self.batchnorm:
            out = self.bn2(out) # [batch_size, conv2c, 5, 5]

        out = out.reshape(out.shape[0], -1) # [batch_size, conv2c*25]
        out = F.relu(self.fc1(out)) # [batch_size, fc1]
        out = self.dropout(out)
        out = F.log_softmax(self.fc2(out), dim = 1) # [batch_size, fc2]
        return out

    def fit(self, trainloader, valloader, lr = 0.001, weight_decay = 1e-5,
        max_epoch = 1, checkpoints_path = ".\\checkpoints"):
        ###########################
        # Your Training Procedure #
        ###########################
        fit_log_file = open("fit_log_file.dat", mode = "w")
        train_batch_losses = []; val_acc = []
    
        loss_fn = nn.NLLLoss(reduction = "mean")
        # Using Adam optimizer
        optimizer = Adam(self.parameters(), lr = lr, weight_decay = weight_decay) 
        batches_per_epoch = len(trainloader)
    
        for epoch in range(max_epoch):
            epoch_loss = 0
            for i, x in enumerate(trainloader):
                optimizer.zero_grad()
                image, label = x
                pred = self(image)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_batch_losses.append((epoch * batches_per_epoch + i, loss.item()))
                self.save_checkpoint(checkpoints_path + "\\model.pt", epoch, loss)
                if (i % 200 == 0):
                    fit_log_file.write("Epoch %d, Batch %d loss: %f"%(epoch, i, loss.item()) + "\n")
                    acc, _prec, _rec, _cm, _mis, _pred = self.evaluation(valloader)
                    val_acc.append((epoch * batches_per_epoch + i, acc))
                    fit_log_file.write("   Accuracy after epoch %d batch %d: %f"%(epoch, i, acc) + "\n")
            fit_log_file.write("\n##### Epoch %d average loss: "%epoch + 
                                    str(epoch_loss/batches_per_epoch) + " #####\n\n")

        return train_batch_losses, val_acc

    @torch.no_grad()
    def evaluation(self, evalloader):
        #############################
        # Your Evaluation Procedure #
        #############################
        conf_mat = npy.zeros((10, 10))
        self.eval()
        prec = npy.zeros(10); rec = npy.zeros(10)
        misclassified = []; predicts = []; numT = 0; numF = 0
        for i, x in enumerate(evalloader):
            image, label = x
            pred = torch.argmax(self(image), dim = 1)
            _T = torch.sum(pred == label).item()
            numT += _T; numF += len(label) - _T
            for j in range(len(label)):
                conf_mat[label[j], pred[j]] += 1
                if label[j] != pred[j]:
                    misclassified.append((image[j], label[j], pred[j]))
                predicts.append(pred[j].item())
        for i in range(0, 10):
            label_cnt = 0; pred_cnt = 0
            for j in range(0, 10): 
                label_cnt += conf_mat[i][j]; pred_cnt += conf_mat[j][i]
            prec[i] = npy.NaN if pred_cnt == 0 else conf_mat[i][i] / pred_cnt
            rec[i] = npy.NaN if label_cnt == 0 else conf_mat[i][i] / label_cnt
        self.train()
        return numT / (numT + numF), prec, rec, conf_mat, misclassified, predicts

    def save_checkpoint(self, path, epoch, loss):
        try:
            optim_state = optimizer.state_dict()
        except:
            optim_state = None
        checkpoint = {
            "model_state_dict": self.state_dict(), "epoch": epoch,
            "loss": loss, "optimizer_state_dict": optim_state
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer = Adam):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["optimizer_state_dict"] is not None:
            self.optimizer = optimizer(self.parameters())
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return epoch, loss

    def eval_handwritten_digits(self, directory):
        images_path = os.listdir(directory)
        images = []
        # read images
        for i in images_path:
            label = int(i.split('_')[1].split('.')[0])
            image = torchvision.io.read_image(os.path.join(directory, i)).float()[0:3,:,:]
            image = image.mean(dim = 0).unsqueeze(0)
            images.append((image, label))
        # Calculate mean and std
        _values = torch.concat([i for i, j in images]).reshape(-1)
        _mean = _values.mean().item()
        _sd = _values.std().item()
        # Data normalization
        mytransform = torchvision.transforms.Normalize((_mean), (_sd))
        for i in range(len(images)):
            img = mytransform(images[i][0])
            images[i] = (img, images[i][1])

        # build custom Dataset
        myevalset = mySet(images)
        # build DataLoader
        myloader = DataLoader(mySet(images), shuffle = False, 
                                                drop_last = False, batch_size = 1)
        # Evaluation
        myacc, myprec, myrec, mycm, mymis, mypred = self.evaluation(myloader)
        return myacc

def load_mnist(mnist_folder_dir):
    # Download MNIST dataset (or load the directly if you have already downloaded them previously)
    if os.path.exists(mnist_folder_dir + "\\MNIST"): _dl = False
    else: _dl = True
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
    trainset_all = torchvision.datasets.MNIST(mnist_folder_dir, train = True, 
                                                        download = _dl, transform = transform)
    testset = torchvision.datasets.MNIST(mnist_folder_dir, train = False, 
                                                        download = _dl, transform = transform)
    return trainset_all, testset

def data_split(trainset_all, valid_size: int, rand_seed: int = 1919810):
    train_size = len(trainset_all) - valid_size
    trainset, valset = random_split(trainset_all, [train_size, valid_size], 
                                    generator=torch.Generator().manual_seed(rand_seed))
    return trainset, valset

def handwritten_classif(model, handwritten_folder_dir: str = ".\\handwritten", 
                                    pred_name: str = "handwritten_prediction.png", batch_size: int = 32):
    images_path = os.listdir(handwritten_folder_dir)
    images = []
    for i in images_path:
        label = int(i.split('_')[1].split('.')[0])
        image = torchvision.io.read_image(os.path.join(
                            handwritten_folder_dir, i)).float()[0:3, :, :]
        image = image.mean(dim = 0).unsqueeze(0)
        images.append((image, label))

    _values = torch.concat([i for i, j in images]).reshape(-1)
    _mean = _values.mean().item()
    _sd = _values.std().item()
    # print("Mean: ",_mean, "  Std:",_sd)
    mytransform = torchvision.transforms.Normalize((_mean), (_sd))

    for i in range(len(images)):
        img = mytransform(images[i][0])
        images[i] = (img, images[i][1]) 
        
    myevalset = mySet(images)
    # print("My dataset size: ", len(myevalset))

    myloader = DataLoader(mySet(images), shuffle = False, drop_last = False, batch_size = batch_size) 
    myacc, myprec, myrec,  mycm, mymis, mypred = model.evaluation(myloader)

    # print("Accuracy on custom dataset: ", myacc)
    # print("Visualizing custom samples")
    plt.figure()
    for i in range(len(myevalset)):
        img, label = myevalset[i]
        pred = mypred[i]
        img = img.squeeze()
        plt.subplot((len(myevalset) + 1) // 5, 5, i + 1)
        plt.imshow(img, cmap = 'gray', interpolation = 'none')
        plt.title("Label: " + str(label) + "\nPredict: " + str(pred))
        plt.axis('off')
    plt.savefig(pred_name, dpi = 300)

def model_evaluation(model_cnn, evalloader, eval_file_name: str = "cnn_eval_file.dat"):
    cnn_acc, cnn_prec, cnn_rec, cnn_cm, cnn_mis, cnn_pred = model_cnn.evaluation(evalloader)
    cnn_cm = pd.DataFrame(cnn_cm, dtype = int)
    cnn_eval_file = open(eval_file_name, mode = "w")
    cnn_eval_file.write("CNN Prediction Accuracy: " + str(cnn_acc) + "\n")
    cnn_eval_file.write("CNN Prediction Confusion Matrix: \n" + str(cnn_cm) + "\n")
    cnn_eval_file.write("CNN Prediction Precision: \n")
    for i in range(0, 10):
        cnn_eval_file.write("Number " + str(i) + ": " + str(cnn_prec[i]) + "\n")
    cnn_eval_file.write("CNN Prediction Recall: \n")
    for i in range(0, 10):
        cnn_eval_file.write("Number " + str(i) + ": " + str(cnn_rec[i]) + "\n")
    return cnn_acc, cnn_prec, cnn_rec, cnn_cm, cnn_mis, cnn_pred

def main():
    os.chdir(sys.path[0])
    mnist_folder_dir = "..\\data"
    trainset_all, testset = load_mnist(mnist_folder_dir)
    trainset, valset = data_split(trainset_all, 10000)

    # hyperparameters
    batch_size = 64
    # batch_size: 32, 64, 128
    # for CNN model (conv2d x2 + fc x2)
    conv1c = 16; conv2c = 32; conv1k = 5; conv2k = 3
    fc1 = 128; fc2 = 10; batchnorm = True; dropout = 0.1
    # for training
    lr = 0.001; weight_decay = 1e-5
    # lr choice: 0.1, 0.01, 0.001, 0.0001

    trainloader = DataLoader(trainset, batch_size, shuffle = True, drop_last = True)
    valloader = DataLoader(valset, batch_size, shuffle = False, drop_last = False)
    testloader = DataLoader(testset, batch_size, shuffle = False, drop_last = False)

    # training and evaluation of validation and test data
    model_cnn = CNN(conv1c, conv2c, conv1k, conv2k, 
                                    fc1, fc2, batchnorm, dropout)
    model_cnn.train()
    train_losses_cnn, val_acc_cnn = model_cnn.fit(trainloader, valloader, lr = lr, 
                                                            weight_decay = weight_decay, max_epoch = 3)
    model_evaluation(model_cnn, valloader, "cnn_val_eval_file.dat")
    cnn_test_acc, cnn_test_prec, cnn_test_rec, \
        cnn_test_cm, cnn_test_mis, cnn_test_pred = model_evaluation(model_cnn, testloader, "cnn_test_eval_file.dat")
    # pd.concat([pd.Series(npy.arange(1, len(cnn_test_pred) + 1), name = "Test Number"),
    #                pd.Series(testset.targets, name = "Test Label"), 
    #                pd.Series(cnn_test_pred, name = "Test Prediction")], 
    #                axis = 1).to_csv("mnist_test_prediction.csv", index = False)
    pd.concat([pd.Series(npy.arange(1, len(cnn_test_pred) + 1)), 
                    pd.Series(testset.targets), pd.Series(cnn_test_pred)], axis = 1).to_csv(
                    "mnist_test_prediction.csv", index = False, header = False)

    # prediction of handwritten data
    handwritten_folder_dir = ".\\handwritten"
    handwritten_inv_folder_dir = ".\\handwritten_inv"
    handwritten_classif(model_cnn, handwritten_folder_dir, "handwritten_prediction.png", batch_size)
    handwritten_classif(model_cnn, handwritten_inv_folder_dir, "handwritten_inv_prediction.png", batch_size)

if __name__ == '__main__':
    main()