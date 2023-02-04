from scipy.stats import norm
from abc import ABCMeta, abstractmethod
import numpy as np

class ac_class(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calc_ac(self, X, model, y_best):
        pass     

class random_value(ac_class):
    def __init__(self):
        pass
    def calc_ac(self, X, model, y_best):
        np.random.seed(0)
        value = np.random.randint(0, 1, len(X)).reshape(-1,1)
        return value, value , value 

class expected_value(ac_class):
    def __init__(self):
        pass
    def calc_ac(self, X, model, y_best):
        mean, std = model.pred_dist(X)
        # ac multiplied by -1 to maximize
        return -1 * mean , mean , std

class PI(ac_class):
    def __init__(self, xi :float = 0):
        # xi can also use 0.01
        self.xi = xi
    def calc_ac(self, X, model, y_best):
        mean, std = model.pred_dist(X)
        z = (y_best - mean - self.xi)/std
        return norm.cdf(z), mean , std

class EI(ac_class):
    def __init__(self, xi :float = 0):
        # xi can also use 0.01
        self.xi = xi
    def calc_ac(self, X, model, y_best):
        mean, std = model.pred_dist(X)
        z = (y_best - mean - self.xi)/std
        return (y_best - mean - self.xi) * norm.cdf(z) + std * norm.pdf(z), mean , std

class LCB(ac_class):
    def __init__(self, ratio :float = 2):
        self.ratio = ratio
    def calc_ac(self, X, model, y_best):
        mean, std = model.pred_dist(X)
        # ac multiplied by -1 to maximize
        return -1 * ((mean  - self.ratio * std)), mean , std
