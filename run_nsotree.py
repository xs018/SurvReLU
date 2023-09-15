import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import numpy as np
import matplotlib.pyplot as plt
from sksurv.datasets import get_x_y

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from models.nostree import CoxNOSTree, CoxCCNOSTree

from utils.data_utils import load_dataset, format_dataset_to_df

if __name__ == "__main__":
    dataset = 'metabric'
    data = load_dataset(dataset)

    random_state = 1234
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)
    num_bootstrap = 100

    if 'simulated' in dataset:
        pass
    else:
        kfold_data = data

        C_index_folds = []
        # C_indinces = []

        for fold_idx, fold_data in enumerate(kfold_data):
            train_data, valid_data = fold_data['train'], fold_data['val']
            train_data_df = format_dataset_to_df(train_data, 'T', 'E')
            X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

            val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
            X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

            get_target = lambda df: (df['T'].values, df['E'].values)
            y_train = get_target(train_data_df)
            y_valid = get_target(val_data_df)

            # print(X_train.shape, X_valid.shape)

            X_train = X_train.values.astype(np.float32)
            X_valid = X_valid.values.astype(np.float32)

            durations_val, events_val = get_target(val_data_df)
            val = (X_valid, y_valid)
            
            # in_features = X_train.shape[1]
            # num_nodes = [44, 44]
            # out_features = 1
            # batch_norm = True
            # dropout = 0.2
            # output_bias = False

            # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
            #                             dropout, output_bias=output_bias)
            input_dim = X_train.shape[1]
            output_dim = 1
            hidden_dim = 1
            depth = 6
            back_n = 1
            drop_type = 'node_dropconnect'
            net_type = 'locally_linear'
            anneal = 'approx'

            model = CoxNOSTree(input_dim=input_dim, 
                                output_dim=output_dim, 
                                hidden_dim=hidden_dim, 
                                num_layer=depth, 
                                num_back_layer=back_n, 
                                dense=True, 
                                drop_type=drop_type, 
                                net_type=net_type, 
                                approx=anneal,
                                p_drop = 0.75,
                                optimizer=tt.optim.Adam)
            batch_size = 128
            model.optimizer.set_lr(0.1)
            epochs = 512
            callbacks = [tt.callbacks.EarlyStopping()]
            verbose = False

            log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
            
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(X_valid)
            ev = EvalSurv(surv, durations_val, events_val, censor_surv='km')
            C_index = ev.concordance_td()
            print(C_index)
            C_index_folds.append(C_index)
        
        c_index_mean = float(np.mean(C_index_folds))
        c_index_std = float(np.std(C_index_folds))
        print(f"C-Index = {c_index_mean:.4f} (+- {c_index_std:4f})")