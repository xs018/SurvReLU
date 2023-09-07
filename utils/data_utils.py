import os
import copy
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset


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
    file_path = "datasets/gaussian_survival_data.h5" if dataset == 'gaussian' else "datasets/linear_survival_data.h5"
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
    assert dataset in ['support', 'metabric', 'flchain']
    file_path = os.path.join("datasets", dataset, dataset + ".csv")
    raw_csv = pd.read_csv(file_path)

    KFold_data = []
    kf = KFold(n_splits=n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(raw_csv)):
        train_data = raw_csv.iloc[train_index]
        val_data = raw_csv.iloc[test_index]

        train_X_mean = train_data.iloc[:, :-2].mean(axis=0)
        train_X_std = train_data.iloc[:, :-2].std(axis=0)

        train_data_norm = copy.deepcopy(train_data)
        train_data_norm.iloc[:, :-2] = (train_data_norm.iloc[:, :-2] - train_X_mean) / train_X_std
        train_data = {}
        train_data['x'] = train_data_norm.iloc[:, :-2].values
        train_data['t'] = train_data_norm.iloc[:, -2].values
        train_data['e'] = train_data_norm.iloc[:, -1].values

        val_data_norm = copy.deepcopy(val_data)
        val_data_norm.iloc[:, :-2] = (val_data_norm.iloc[:, :-2] - train_X_mean) / train_X_std
        val_data = {}
        val_data['x'] = val_data_norm.iloc[:, :-2].values
        val_data['t'] = val_data_norm.iloc[:, -2].values
        val_data['e'] = val_data_norm.iloc[:, -1].values
        
        KFold_data.append({'train': train_data, 'val': val_data})

    return KFold_data


def load_dataset(dataset):
    if dataset == 'simulated_linear':
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_simulated_data('linear')
        data = {'train': train_data, 'valid': valid_data, 'test':test_data, 'viz':viz_data, 'viz_unnorm':viz_data_unnorm}
    elif dataset == 'simulated_gaussian':
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_simulated_data('gaussian')
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
    load_real_data('flchain')