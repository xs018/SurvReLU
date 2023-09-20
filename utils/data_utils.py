import os
import copy
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
# from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pycox import datasets
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong

import torch
from torch.utils.data import Dataset
import torchtuples as tt


class SimpleDataset(Dataset):
    '''
    Assuming X and y are numpy arrays and 
     with X.shape = (n_samples, n_features) 
          y.shape = (n_samples,)
    '''
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        #data = np.array(data).astype(np.float32)
        if self.y is not None:
            return dict(input=data, label=self.y[i])
        else:
            return dict(input=data)
        

class FastTensorDataLoader():
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, tensor_names, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param tensor_names: name of tensors (for feed_dict)
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.tensor_names = tensor_names

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = {}
        for k in range(len(self.tensor_names)):
            batch.update({self.tensor_names[k]: self.tensors[k][self.i:self.i+self.batch_size]})
        self.i += self.batch_size
        return batch
        
    def __len__(self):
        return self.n_batches

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def format_dataset_to_df(dataset, duration_col, event_col, trt_idx = None):
    xdf = pd.DataFrame(dataset['x'])
    if trt_idx is not None:
        xdf = xdf.rename(columns={trt_idx : 'treat'})

    dt = pd.DataFrame(dataset['t'], columns=[duration_col])
    censor = pd.DataFrame(dataset['e'], columns=[event_col])
    cdf = pd.concat([xdf, dt, censor], axis=1)
    return cdf

'''standardize_dataset function is from utils_jared.py'''
def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds


def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort training data for accurate partial likelihood calculation.
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    #return x, {'e': e, 't': t} this is for parse_data(x, label); see the third line in the parse_data function. 
    #return {'x': x, 'e': e, 't': t}
    return x, e, t


def load_cox_gaussian(file_path="datasets/gaussian_survival_data.h5", reverse_duration=False):
    # Load dataset
    datasets = load_datasets(file_path)

    # Standardize
    train_data = datasets['train']
    norm_vals = {
                'mean' : datasets['train']['x'].mean(axis=0),
                'std'  : datasets['train']['x'].std(axis=0)
            }
    test_data = datasets['test']

    train_data = standardize_dataset(datasets['train'], norm_vals['mean'], norm_vals['std'])
    valid_data = standardize_dataset(datasets['valid'], norm_vals['mean'], norm_vals['std'])
    test_data = standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
    viz_data = standardize_dataset(datasets['viz'], norm_vals['mean'], norm_vals['std'])

    if reverse_duration:
        train_X = train_data['x']
        train_y = {'e': train_data['e'], 't': train_data['t']}
        valid_X = valid_data['x']
        valid_y = {'e': valid_data['e'], 't': valid_data['t']}
        test_X = test_data['x']
        test_y = {'e': test_data['e'], 't': test_data['t']}
        viz_X = viz_data['x']
        viz_y = {'e': viz_data['e'], 't': viz_data['t']}

        # Sort training data for accurate partial likelihood calculation
        train_data={}
        train_data['x'], train_data['e'], train_data['t'] = prepare_data(train_X, train_y)
        train_data['ties'] = 'noties'

        valid_data={}
        valid_data['x'], valid_data['e'], valid_data['t'] = prepare_data(valid_X, valid_y)
        valid_data['ties'] = 'noties'

        test_data = {}
        test_data['x'], test_data['e'], test_data['t'] = prepare_data(test_X, test_y)
        test_data['ties'] = 'noties'

        viz_data = {}
        viz_data['x'], viz_data['e'], viz_data['t'] = prepare_data(viz_X, viz_y)
        viz_data['ties'] = 'noties'

    return train_data, valid_data, test_data, viz_data, datasets['viz']


def load_simulated_data(dataset="linear", reverse_duration=False):
    # Load dataset
    if dataset == 'linear':
        file_path = "datasets/linear_survival_data.h5"
    elif dataset == 'linear_2d':
        file_path = "datasets/linear_survival_data_2d.h5" 
    elif dataset == 'gaussian':
        file_path = "datasets/gaussian_survival_data.h5"
    elif dataset == 'gaussian_2d':
        file_path = "datasets/gaussian_survival_data_2d.h5" 

    datasets = load_datasets(file_path)

    # Standardize
    train_data = datasets['train']
    norm_vals = {
                'mean' : datasets['train']['x'].mean(axis=0),
                'std'  : datasets['train']['x'].std(axis=0)
            }
    test_data = datasets['test']

    train_data = standardize_dataset(datasets['train'], norm_vals['mean'], norm_vals['std'])
    valid_data = standardize_dataset(datasets['valid'], norm_vals['mean'], norm_vals['std'])
    test_data = standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
    viz_data = standardize_dataset(datasets['viz'], norm_vals['mean'], norm_vals['std'])

    if reverse_duration:
        train_X = train_data['x']
        train_y = {'e': train_data['e'], 't': train_data['t']}
        valid_X = valid_data['x']
        valid_y = {'e': valid_data['e'], 't': valid_data['t']}
        test_X = test_data['x']
        test_y = {'e': test_data['e'], 't': test_data['t']}
        viz_X = viz_data['x']
        viz_y = {'e': viz_data['e'], 't': viz_data['t']}

        # Sort training data for accurate partial likelihood calculation
        train_data={}
        train_data['x'], train_data['e'], train_data['t'] = prepare_data(train_X, train_y)
        train_data['ties'] = 'noties'
 
        valid_data={}
        valid_data['x'], valid_data['e'], valid_data['t'] = prepare_data(valid_X, valid_y)
        valid_data['ties'] = 'noties'

        test_data = {}
        test_data['x'], test_data['e'], test_data['t'] = prepare_data(test_X, test_y)
        test_data['ties'] = 'noties'

        viz_data = {}
        viz_data['x'], viz_data['e'], viz_data['t'] = prepare_data(viz_X, viz_y)
        viz_data['ties'] = 'noties'

    return train_data, valid_data, test_data, viz_data, datasets['viz']

def load_real_data(dataset, n_folds=5):
    if dataset == 'support':
        df = datasets.support.read_df()
        cols_standardize =  ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        cols_leave = ['x1', 'x4', 'x5']
        cols_categorical =  ['x2', 'x3', 'x6']
    elif dataset == 'metabric':
        df = datasets.metabric.read_df()
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        cols_categorical = []
    elif dataset == 'flchain':
        df = datasets.flchain.read_df()
        cols_standardize = ['age', 'kappa', 'lambda', 'creatinine']
        cols_leave = ['sex', 'mgus']
        cols_categorical = ['sample.yr', 'flc.grp']
    elif dataset == 'gbsg':
        df = datasets.gbsg.read_df()
        cols_standardize = ['x3', 'x4', 'x5']
        cols_leave = ['x0', 'x1', 'x2']
        cols_categorical = ['x6']
    else:
        raise Exception()
    
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize + leave)
    x_mapper_long = DataFrameMapper(categorical)  # we need a separate mapper to ensure the data type 'int64'

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))

    KFold_data = []
    kf = KFold(n_splits=n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        df_train = df.iloc[train_index]
        df_val = df.iloc[test_index]

        x_train_float = x_mapper_float.fit_transform(df_train)
        x_valid_float = x_mapper_float.transform(df_val)

        if len(categorical) > 0:
            x_train_long = x_mapper_long.fit_transform(df_train)
            x_valid_long = x_mapper_long.transform(df_val)

            x_train = np.concatenate([x_train_float, x_train_long], axis=1)
            x_valid = np.concatenate([x_valid_float, x_valid_long], axis=1)
        else:
            x_train = x_train_float
            x_valid = x_valid_float

        # print(x_train.shape, x_valid.shape)

        # x_train = x_fit_transform(df_train)
        # x_train = np.concatenate(x_train, axis=-1)
        # x_valid = x_transform(df_val)
        # x_valid = np.concatenate(x_valid, axis=-1)

        y_train = df_train.iloc[:, -2:].values
        y_valid = df_val.iloc[:, -2:].values

        train_data = {}
        train_data['x'] = x_train
        train_data['t'] = y_train[:, 0]
        train_data['e'] = y_train[:, 1]

        val_data = {}
        val_data['x'] = x_valid
        val_data['t'] = y_valid[:, 0]
        val_data['e'] = y_valid[:, 1]
        
        KFold_data.append({'train': train_data, 'val': val_data})

    return KFold_data

# def load_real_data(dataset, n_folds=5):
#     assert dataset in ['support', 'metabric', 'flchain']
#     file_path = os.path.join("datasets", dataset, dataset + ".csv")
#     raw_csv = pd.read_csv(file_path)

#     if dataset == 'support':
#         onehot_variables = ['x2', 'x3', 'x6'] 
#         standardize_variables = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
#         # leave_variables = ['x1', 'x4', 'x5']
#     elif dataset == 'metabric':
#         onehot_variables = [] 
#         standardize_variables = ['x0', 'x1', 'x2', 'x3', 'x8']
#         # leave_variables = ['x4', 'x5', 'x6', 'x7']
#     elif dataset == 'flchain':
#         onehot_variables = ['sample.yr', 'flc.grp'] 
#         standardize_variables = ['age', 'kappa', 'lambda', 'creatinine']
#         # leave_variables = ['sex', 'mgus']

#     if len(onehot_variables) > 0:
#         encoder = OneHotEncoder()
#         encoded_variables = pd.DataFrame(encoder.fit_transform(raw_csv[onehot_variables]).toarray())
        
#         raw_csv = raw_csv.drop(columns=onehot_variables)
#         raw_csv = pd.concat([encoded_variables, raw_csv], axis=1)

#         # print(raw_csv)

#         # X_train_t = np.concatenate([X_train_t, encoder.fit_transform(X_train[onehot_variables]).toarray()], axis=1)
#         # X_valid_t = np.concatenate([X_valid_t, encoder.transform(X_valid[onehot_variables]).toarray()], axis=1)

#     KFold_data = []
#     kf = KFold(n_splits=n_folds)
#     for i, (train_index, test_index) in enumerate(kf.split(raw_csv)):
#         train_data = raw_csv.iloc[train_index]
#         val_data = raw_csv.iloc[test_index]

#         X_train, y_train = train_data.iloc[:, :-2], train_data.iloc[:, -2:]
#         X_valid, y_valid = val_data.iloc[:, :-2], val_data.iloc[:, -2:]

#         # X_train_t = X_train[leave_variables].values
#         # X_valid_t = X_valid[leave_variables].values
        
#         X_train_t = copy.deepcopy(X_train)
#         X_valid_t = copy.deepcopy(X_valid)

#         if len(standardize_variables) > 0:
#             standardizer = StandardScaler()
#             # X_train_t = np.concatenate([X_train_t, standardizer.fit_transform(X_train[standardize_variables])], axis=1)
#             # X_valid_t = np.concatenate([X_valid_t, standardizer.transform(X_valid[standardize_variables])], axis=1)
#             X_train_t[standardize_variables] = standardizer.fit_transform(X_train[standardize_variables])
#             X_valid_t[standardize_variables] = standardizer.transform(X_valid[standardize_variables])

#         X_train_t, X_valid_t = X_train_t.values, X_valid_t.values

#         train_data = {}
#         train_data['x'] = X_train_t
#         train_data['t'] = y_train.values[:, 0]
#         train_data['e'] = y_train.values[:, 1]

#         val_data = {}
#         val_data['x'] = X_valid_t
#         val_data['t'] = y_valid.values[:, 0]
#         val_data['e'] = y_valid.values[:, 1]
        
#         KFold_data.append({'train': train_data, 'val': val_data})

#     return KFold_data


def load_dataset(dataset):
    if dataset == 'simulated_linear':
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_simulated_data('linear')
        data = {'train': train_data, 'valid': valid_data, 'test':test_data, 'viz':viz_data, 'viz_unnorm':viz_data_unnorm}
    elif dataset == 'simulated_linear_2d':
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_simulated_data('linear_2d')
        data = {'train': train_data, 'valid': valid_data, 'test':test_data, 'viz':viz_data, 'viz_unnorm':viz_data_unnorm}
    elif dataset == 'simulated_gaussian':
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_simulated_data('gaussian')
        data = {'train': train_data, 'valid': valid_data, 'test':test_data, 'viz':viz_data, 'viz_unnorm':viz_data_unnorm}
    elif dataset == 'simulated_gaussian_2d':
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_simulated_data('gaussian_2d')
        data = {'train': train_data, 'valid': valid_data, 'test':test_data, 'viz':viz_data, 'viz_unnorm':viz_data_unnorm}
    else:
        data = load_real_data(dataset)

    return data
    
# class CoxGaussianDataset(Dataset):
#     def __init__(self, dataroot="datasets/gaussian_survival_data.h5"):
#         super().__init__()
#         self._dataset = load_datasets(dataroot)

#     def __len__(self):
#         pass

#     def __getitem__(self, idx):
#         pass

if __name__ == "__main__":
    load_real_data('gbsg')