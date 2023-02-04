import dataclasses
from typing import List,Union
import pandas as pd
from importlib import resources
import numpy as np

DATA_MODULE = "material_bench.datasets"

@dataclasses.dataclass
class material_dataset_class:
    dataset_name: Union[str,None] = None
    x_name_list: Union[List[str],None] = None
    y_name_list: Union[List[str],None] = None
    data: Union[pd.DataFrame,None] = None
    direction_adjustment:bool = False

def load_exp_dataset(dataset_name :str ='CrossedBarrel', pretreatment :bool = True, conv_min_dir :bool = True,data_module :str= DATA_MODULE) :

    # dataset_name_list : 'Crossed barrel', 'Perovskite', 'AgNP', 'P3HT', 'AutoAM','Concrete','Thermoelectrics'
    max_dir_datset_list = ['CrossedBarrel', 'P3HT', 'AutoAM','Concrete','Thermoelectrics']


    # load csv file
    file_name = dataset_name + '_dataset.csv'
    with resources.open_text(data_module, file_name) as csv_file:
        csv_df = pd.read_csv(csv_file)

    # pretreatment 1
    if pretreatment is True:
        if dataset_name == 'Thermoelectrics':
            drop = ['index', 'nelements', 'n', 'p', 'nsites', 'direct', 'indirect']
            csv_df = csv_df.drop(drop, axis=1)

    # load data set
    dataset_class = material_dataset_class()
    dataset_class.dataset_name = dataset_name
    dataset_class.x_name_list = list(csv_df.columns[:-1])
    dataset_class.y_name_list = list(csv_df.columns[-1:])
    if (conv_min_dir is True) and (dataset_name in max_dir_datset_list):
        csv_df[dataset_class.y_name_list] = -1*csv_df[dataset_class.y_name_list]

        dataset_class.direction_adjustment = True

    # pretreatment 2
    if pretreatment is True:
        csv_df = csv_df.groupby(dataset_class.x_name_list)[dataset_class.y_name_list].agg(lambda x: x.unique().mean())
        csv_df = csv_df.reset_index()

    dataset_class.data = csv_df      

    return dataset_class


def load_model_dataset(dataset_name :str ='StyblinskiTang', min_value:int =-5, max_value :int =5,n :int=100000, d :int= 30, random_state :int =0) :

    # dataset_name_list : 'StyblinskiTang', 'RosenBrock'

    # create X 
    x_array = _rand_matrix(min_value, max_value, n, d, random_state)

    # calc y
    if dataset_name == 'StyblinskiTang':
        y_array = ((x_array**4) - (16 * x_array**2) + (5*x_array)).sum(axis=1) * 0.5
    elif dataset_name == 'RosenBrock':
        y_array = (100*(x_array[:,1:] - x_array[:,:-1]**2)**2 + (x_array[:,:-1] - 1)**2).sum(axis=1)
    else:
        raise ValueError

    # load data set
    dataset_class = material_dataset_class()
    dataset_class.dataset_name = dataset_name + '_' + str(d)
    dataset_class.x_name_list = [f'x_{i}' for i in range(x_array.shape[1])]
    dataset_class.y_name_list = [f'y_{dataset_name}']
    dataset_class.data = pd.DataFrame(np.hstack((x_array,y_array.reshape(-1,1))),columns=dataset_class.x_name_list + dataset_class.y_name_list)     

    return dataset_class

def _rand_matrix(min_value, max_value, n, d, random_state):
  np.random.seed(random_state)
  random_array = np.random.randint(min_value*100, max_value*100,(n,d))/100
  unique_random_array = np.unique(random_array, axis=0)
  if len(random_array) < d:
    print('The number of samples is less than the set number of samples due to overlapping values')
  return unique_random_array