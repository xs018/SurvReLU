from stg import STG
import stg.utils as utils
import numpy as np
import torch
import time

datasets = utils.load_cox_gaussian_data()

train_data = datasets['train']
norm_vals = {
        'mean' : datasets['train']['x'].mean(axis=0),
        'std'  : datasets['train']['x'].std(axis=0)
    }
test_data = datasets['test']

# standardize
train_data = utils.standardize_dataset(datasets['train'], norm_vals['mean'], norm_vals['std'])
valid_data = utils.standardize_dataset(datasets['valid'], norm_vals['mean'], norm_vals['std'])
test_data = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])

train_X = train_data['x']
train_y = {'e': train_data['e'], 't': train_data['t']}
valid_X = valid_data['x']
valid_y = {'e': valid_data['e'], 't': valid_data['t']}
test_X = test_data['x']
test_y = {'e': test_data['e'], 't': test_data['t']}

train_data={}
train_data['X'], train_data['E'], train_data['T'] = utils.prepare_data(train_X, train_y)
train_data['ties'] = 'noties'

valid_data={}
valid_data['X'], valid_data['E'], valid_data['T'] = utils.prepare_data(valid_X, valid_y)
valid_data['ties'] = 'noties'

test_data = {}
test_data['X'], test_data['E'], test_data['T'] = utils.prepare_data(test_X, test_y)
test_data['ties'] = 'noties'