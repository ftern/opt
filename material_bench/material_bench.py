import numpy as np
import math
import time
import random
from sklearn import preprocessing
import pickle
import warnings
import copy


class benchmarking_mi_opt:
    def __init__(self):
        self.index_collection = []
        self.mat_dataset = None
        self.y_best_trend_collection = []
        self.y_pred_mean_collection = []
        self.y_pred_std_collection = []
        self.total_time = None
        self.batch_size = None

    def calc_pool_opt(self,model, mat_dataset, ac_class, sample_select_type :str= 'random', n_exp_initial: int = 10, n_ensemble: int = 5, result_save_name :str="result", iterations :int= -1):
        
        # calc time
        start_time = time.time()

        self.mat_dataset = mat_dataset
        x_all = mat_dataset.data[mat_dataset.x_name_list].values
        y_all = mat_dataset.data[mat_dataset.y_name_list].values.ravel()
        all_data_num = mat_dataset.data[mat_dataset.x_name_list].shape[0]

        np.random.seed(0)
        seed_list = list(np.random.randint(0, 10000, (n_ensemble)))

        for ensemble_num in range(n_ensemble):
            
            print(f'ensumble num = {ensemble_num}, initializing seed = {str(seed_list[ensemble_num])}')
            random.seed(seed_list[ensemble_num])

            # index_learn_pool is the pool of candidates to be examined
            # index_ is the list of candidates we have already observed
            index_learn_pool = list(np.arange(all_data_num))
            index_observed = random.sample(index_learn_pool, n_exp_initial)
            index_learn_pool = [i for i in index_learn_pool if i not in index_observed]
            
            # for each of the the rest of (N - n_initial) learning cycles
            # this for loop ends when all candidates in pool are observed 
            y_best_trend_list = []
            y_pred_mean_list = []
            y_std_mean_list = []

            if iterations == -1:
                iterations = len(index_learn_pool)
            elif iterations > len(index_learn_pool):
                iterations = len(index_learn_pool)
            for _ in np.arange(iterations):
                next_index = None
                x_observed = x_all[index_observed,:]
                y_observed = y_all[index_observed]

                x_scaler = preprocessing.StandardScaler()
                y_scaler = preprocessing.StandardScaler()            
                x_train_std = x_scaler.fit_transform(x_observed)
                x_all_std = x_scaler.transform(x_all)
                y_observed_std = y_scaler.fit_transform(y_observed.reshape(-1,1)).ravel()
                y_best_std = np.min(y_observed_std)
                model.fit(x_train_std, y_observed_std)

                # by evaluating acquisition function values at candidates remaining in pool
                # we choose candidate with larger acquisition function value to be observed next    
                ac_value_array, y_pred_mean_array, y_pred_std_array = ac_class.calc_ac(x_all_std, model, y_best_std)

                next_index = self._sample_select(ac_value_array, x_all_std,index_learn_pool, index_observed, sample_select_type)

                index_observed.append(next_index)
                index_learn_pool.remove(next_index)
                
                y_pred_mean = y_scaler.inverse_transform(y_pred_mean_array[next_index].reshape(-1,1))
                y_std_mean = y_scaler.inverse_transform(y_pred_std_array[next_index].reshape(-1,1))

                y_best_trend_list.append(np.min(y_observed))
                y_pred_mean_list.append(y_pred_mean[0][0])
                y_std_mean_list.append(y_std_mean[0][0])
            else:
                y_observed = y_all[index_observed]
                y_best_trend_list.append(np.min(y_observed))
                
                self.index_collection.append(index_observed)
                self.y_best_trend_collection.append(y_best_trend_list)
                self.y_pred_mean_collection.append(y_pred_mean_list)
                self.y_pred_std_collection.append(y_std_mean_list)
            
        self.total_time = time.time() - start_time
        
        if result_save_name is not None:
            with open(result_save_name, 'wb') as f:
                pickle.dump(self, f)

        return self


    def calc_avg_performance(self):
        result_dic = {'y_best_trend':self._calc_ave_array(self.y_best_trend_collection),
                    'y_pred_mean':self._calc_ave_array(self.y_pred_mean_collection),
                    'y_pred_std':self._calc_ave_array(self.y_pred_std_collection)
                    }

        return result_dic

    def _calc_ave_array(self, result_list):

        avg_array = np.zeros(len(result_list[0])) 
        for one_result in result_list:
            avg_array += np.array(one_result)
        avg_array = avg_array / len(result_list)
        return avg_array


    def _sample_select(self,ac_value_array, x_all_std,index_learn_pool, index_observed, sample_select_type):

        max_ac = np.max(ac_value_array[index_learn_pool])
        max_pool_index = np.where(ac_value_array == max_ac)[0].tolist()
        max_pool_index = [i for i in max_pool_index if i not in index_observed]

        if sample_select_type == 'random': 
            next_index = random.sample(max_pool_index,1)[0]
        elif sample_select_type == 'sequential':
            next_index = max_pool_index[-1]
        else:
            raise ValueError('unexpected sample_select_type')

        #print(f'next_index:{next_index},max pool num/ pool/ all = {len(max_pool_index)}/{len(index_learn_pool)}/{len(x_all_std)}')

        return next_index 
