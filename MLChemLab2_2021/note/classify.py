import os
import random
import numpy as np
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_data():
    if os.path.exists('X.npy'):
        return np.load('X.npy'), np.load('Y.npy')
    with open('train_valid.csv', 'r') as reader:
        lines = reader.readlines()[1 :]
    X = []
    Y = []
    for line in lines:
        feats = line.strip().split(',')
        x = [
            {'Male': 0, 'Female': 1}[feats[1]],  # Gender
            int(feats[2]),  # Age
            int(feats[3]),  # License
            int(feats[4]),  # Retion Code
            int(feats[5]),  # Insured
            {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}[feats[6]],  # Vehicle Age
            {'No': 0, 'Yes': 1}[feats[7]],  # Vehicle Damage
            int(feats[8]),  # Annual Premium
            int(feats[9]),  # Policy Sales Channel
            int(feats[10]),  # Vintage
        ]
        y = int(feats[-1])
        X.append(x)
        Y.append(y)
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    return X, Y


def evaluate(pred, gt):
    '''
    Params:
        pred: np.array, shape = (N, ), unique = [0, 1]
        gt: np.array, shape = (N, ), unique = [0, 1]
    Return:
        f1: float
        precision: float
        recall: float
        accuracy: float
    '''
    acc = np.mean(pred == gt)
    tp = np.sum(pred * gt)
    pp = np.sum(pred)
    gp = np.sum(gt)
    prec = tp / pp
    reca = tp / gp
    f1 = 2 * prec * reca / (prec + reca)
    return {
        'f1': f1,
        'precision': prec,
        'recall': reca,
        'accuracy': acc,
    }



def main():
    # seed
    seed = 233
    random.seed(seed)
    np.random.seed(seed)
    # load data, normalization
    X, Y = load_data()  # X.shape = (N, C), Y.shape = (N, )
    X = (X - np.mean(X, axis=0).reshape(1, -1)) / (np.std(X, axis=0).reshape(1, -1) + 1e-12)
    #
    X = X.tolist()
    random.shuffle(X)
    valid_percent = 0.25
    valid_offset = round(len(X) * valid_percent)
    valid_X = np.array(X[: valid_offset])
    valid_Y = np.array(Y[: valid_offset])
    train_X = np.array(X[valid_offset :])
    train_Y = np.array(Y[valid_offset :])
    #
    train_X_1 = train_X[train_Y == 1].copy()
    train_X_0 = train_X[train_Y == 0].copy()
    train_X_0 = np.array(random.sample(train_X_0.tolist(), train_X_1.shape[0]))
    train_X = np.concatenate([train_X_0, train_X_1], axis=0)
    train_Y = np.concatenate([np.zeros(train_X_0.shape[0]), np.ones(train_X_1.shape[0])]).astype(np.int32)
    #
    # model = DecisionTreeClassifier(random_state=0)
    model = GaussianNB()
    model.fit(train_X, train_Y)
    train_pred = model.predict(train_X)
    print('Train')
    print(evaluate(train_pred, train_Y))
    valid_pred = model.predict(valid_X)
    print('Valid')
    print(evaluate(valid_pred, valid_Y))


if __name__ == '__main__':
    main()