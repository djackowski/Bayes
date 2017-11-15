import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator
from tasks_students.Probability import Probability
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
                # print counter
                # self.probabies.append(Probability(X[k][index], y[k], counter, self.classes_[i]))
                self.model[self.classes_[i]][y[k]][X[k][index]] = counter
                # self.model[self.classes_[i]][] =
                # add to dict
                counter = 0
                index = index + 1
            # print "\n\n"
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
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
