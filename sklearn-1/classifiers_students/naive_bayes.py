import numpy as np
import math
from sklearn.base import BaseEstimator
from collections import defaultdict


class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = [0, 1, 2, 3]
        self.model = defaultdict(lambda: defaultdict(dict))
        self.yesGrypaProb = 0
        self.noGrypaProb = 0
        self.probabies = []

    def fit(self, X, y):
        index = 0
        counter = 0
        yesGrypa = np.count_nonzero(y)
        noGrypa = len(y) - yesGrypa
        self.yesGrypaProb = yesGrypa / float(len(y))
        self.noGrypaProb = noGrypa / float(len(y))

        for k in range(0, len(y)):
            for i in range(0, len(X[0])):
                for j in range(0, len(y)):
                    tempX = X[k][index]
                    tempY = y[k]
                    if (tempX == X[j][index]) and (tempY == y[j]):
                        counter = counter + 1
                if tempY == 0:
                    counter = counter / float(noGrypa)
                else:
                    counter = counter / float(yesGrypa)
                self.model[self.classes_[i]][y[k]][X[k][index]] = counter
                counter = 0
                index = index + 1
            index = 0

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        eventual = []
        for i in range(0, len(X)):
            num = self.yesGrypaProb
            denom = self.noGrypaProb
            for j in range(0, len(X[i])):
                num *= self.model[j][1][X[i][j]]
                denom *= self.model[j][0][X[i][j]]
            result = num / (float(denom + num))
            if result > 0.5:
                eventual.append(1)
            else:
                eventual.append(0)
        return eventual

class NaiveBayesGaussian:
    def __init__(self):
        self.deviation = defaultdict(list)
        self.mean = defaultdict(list)
        self.classes = []
        self.classStartIndexes = []

    def fit(self, X, y):
        self.classes, self.classStartIndexes = np.unique(y, return_index=True)
        self.classStartIndexes = np.append(self.classStartIndexes, len(y))

        for i in self.classes:
            Xclasses = []
            for j in range(self.classStartIndexes[i], self.classStartIndexes[i + 1]):
                Xclasses.append(X[j])
            self.deviation[i].append(np.std(Xclasses, axis=0))
            self.mean[i].append(np.mean(Xclasses, axis=0))

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        classesLength = defaultdict(dict)
        onesProbs = defaultdict(dict)
        allProbs = defaultdict(dict)
        eventual = []

        for i in self.classes:
            classesLength[i] = self.classStartIndexes[i + 1] - self.classStartIndexes[i]

        for i in range(0, len(X)):
            allProbs[i] = 0

        for i in self.classes:
            onesProbs[i] = 1

        for k in range(0, len(X)):
            index = 0
            tempProbs = defaultdict(dict)
            for i in X[k]:
                for j in self.classes:
                    allY = self.classStartIndexes[len(self.classStartIndexes) - 1]
                    currentDeviation = self.deviation[j][0][index]
                    stdSquare = self.deviation[j][0][index] * math.sqrt(2 * math.pi)
                    currentMean = self.mean[j][0][index]
                    rowMean = i - currentMean
                    exp = pow(math.e, -math.pow(rowMean, 2) / float(2 * math.pow(
                        currentDeviation, 2)))
                    pY = float((classesLength[j]) / float(allY))
                    if j not in tempProbs.keys():
                        tempProbs[j] = pY * (1 / float((stdSquare))) * exp
                    else:
                        tempProbs[j] *= (1 / float((stdSquare))) * exp
                    allProbs[k] += tempProbs[j]
                index += 1

            for l in self.classes:
                tempProbs[l] /= allProbs[k]
            eventual.append(max(tempProbs, key=tempProbs.get))
        return eventual


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
