import torch
import torchtuples as tt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from models.nostree import CoxNOSTree



np.random.seed(1234)
_ = torch.manual_seed(123)

input_dim = 9
output_dim = 1
hidden_dim = 1
depth = 15
back_n = 0
drop_type = 'node_dropconnect'
net_type = 'locally_constant'
anneal = 'approx'

model = CoxNOSTree(
    input_dim=input_dim, 
    output_dim=output_dim, 
    hidden_dim=hidden_dim, 
    num_layer=depth, 
    num_back_layer=back_n, 
    dense=True, 
    drop_type=drop_type, 
    net_type=net_type, 
    approx=anneal,
    optimizer=tt.optim.Adam
)

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print(x_train.shape)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val

batch_size = 512

model.optimizer.set_lr(0.1)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test, censor_surv=None)
print(ev.concordance_td())