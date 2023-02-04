from abc import ABCMeta, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.utils import check_X_y,check_array
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random

class model_y_std(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def pred_dist(self, X):
        pass 

class GPRegression_bo(model_y_std):
    def __init__(self):
        pass

    def fit(self, X,Y):
        X, Y = check_X_y(X, Y, accept_sparse=True, y_numeric=True)
        kernel =  ConstantKernel() * RBF()+ WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
        self.model = GaussianProcessRegressor(kernel, alpha=0).fit(X, Y)
        return self

    def pred_dist(self, X):
        X = check_array(X, accept_sparse=True)
        pred_value = self.model.predict(X,return_std =True)
        y_test, std= pred_value[0], pred_value[1]

        return y_test,std


class RandomForestRegressor_bo(model_y_std):
    def __init__(self,random_state :int = 2147483647, n_estimators :int= 100,min_variance:float =0.):
        self.n_est = n_estimators
        self.random_state_ = random_state
        self.min_variance = min_variance

    def fit(self, X,Y):
        X, Y = check_X_y(X, Y, accept_sparse=True, y_numeric=True)
        self.model = RandomForestRegressor(random_state = self.random_state_,n_estimators = self.n_est).fit(X, Y)
        #print(f'score :{self.model.score(X, Y)}')
        return self

    def pred_dist(self, X,sampling_size=None):
        X = check_array(X, accept_sparse=True)
        if sampling_size is None:
            sampling_size = self.n_est
        y_test = self.model.predict(X)
        std = np.zeros(len(X))
        random.seed(self.random_state_)
        estimators = random.sample(self.model.estimators_, sampling_size)

        for estimator in estimators:
            var_estimator = estimator.tree_.impurity[estimator.apply(X)]

            var_estimator[var_estimator < self.min_variance] = self.min_variance
            mean_tree = estimator.predict(X)
            y_test += mean_tree
            std += var_estimator + mean_tree ** 2

        y_test /= sampling_size
        std /= sampling_size
        std -= y_test ** 2.0
        std[std < 0.0] = 0.0
        std = std ** 0.5


        return y_test,std

class BwOForestRegressor_bo(model_y_std):
    def __init__(self,random_state :int = 2147483647, n_estimators :int= 100,min_variance:float =0.,max_features='sqrt',bootstrap=True,rate_oversampling = 4.0,
        num_duplicates_coeff = 4.0):
        self.n_est = n_estimators
        self.random_state_ = random_state
        self.min_variance = min_variance
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.rate_oversampling = rate_oversampling
        self.num_duplicates = int(self.rate_oversampling * num_duplicates_coeff)

    def fit(self, X,Y):
        X, Y = check_X_y(X, Y, accept_sparse=True, y_numeric=True)
        max_samples= float(self.rate_oversampling / self.num_duplicates) # float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0.0, 1.0].

        X = np.tile(X, (self.num_duplicates, 1))
        Y = np.tile(Y, (self.num_duplicates, ))        

        self.model = ExtraTreesRegressor(
            n_estimators=self.n_est,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_state_,
            max_samples= max_samples,
            ).fit(X, Y)

        #print(f'score :{self.model.score(X, Y)}')
        return self

    def pred_dist(self, X,sampling_size=None):
        X = check_array(X, accept_sparse=True)
        if sampling_size is None:
            sampling_size = self.n_est
        y_test = 0
        std = np.zeros(len(X))
        random.seed(self.random_state_)
        estimators = random.sample(self.model.estimators_, sampling_size)

        for estimator in estimators:
            var_estimator = estimator.tree_.impurity[estimator.apply(X)]

            var_estimator[var_estimator < self.min_variance] = self.min_variance
            mean_tree = estimator.predict(X)
            y_test += mean_tree
            std += var_estimator + mean_tree ** 2

        y_test /= sampling_size
        std /= sampling_size
        std -= y_test ** 2.0
        std[std < 0.0] = 0.0
        std = std ** 0.5
        

        return y_test,std

class dummy_model(model_y_std):
    def __init__(self):
        pass

    def fit(self, X,Y):
        return self

    def pred_dist(self, X):
        y_test = np.ones(len(X)).reshape(-1,1)
        std = np.ones(len(X)).reshape(-1,1)
        return y_test,std



