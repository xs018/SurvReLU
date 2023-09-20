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
    dataset = 'simulated_gaussian'
    data = load_dataset(dataset)

    random_state = 1234
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)

    if 'simulated' in dataset:
        num_bootstrap = 100
        vis = False
        save_file = f"{dataset}_Cox-MLP_NOSTree.npy"

        hidden_dim = 1
        depth = 12
        back_n =  0
        drop_type = 'node_dropconnect'
        p_dropout = 0.0
        net_type = 'locally_constant'
        anneal = 'approx'
        batch_size = 512
        epochs = 100
        lr = 0.2
        momentum = 0.95

        train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
        train_data_df = format_dataset_to_df(train_data, 'T', 'E')
        X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)
        val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
        X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)
        test_data_df = format_dataset_to_df(test_data, 'T', 'E')
        X_test, y_test = get_x_y(test_data_df, attr_labels=['E', 'T'], pos_label=1)

        get_target = lambda df: (df['T'].values, df['E'].values)
        y_train = get_target(train_data_df)
        y_valid = get_target(val_data_df)
        y_test = get_target(test_data_df)

        X_train = X_train.values.astype(np.float32)
        X_valid = X_valid.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)

        durations_test, events_test = get_target(test_data_df)
        val = (X_valid, y_valid)

        input_dim = X_train.shape[1]
        output_dim = 1
    
        model = CoxNOSTree(input_dim=input_dim, 
                                output_dim=output_dim, 
                                hidden_dim=hidden_dim, 
                                num_layer=depth, 
                                num_back_layer=back_n, 
                                dense=True, 
                                drop_type=drop_type, 
                                net_type=net_type, 
                                approx=anneal,
                                p_drop = p_dropout,
                                L1_reg = 2e-3,
                                optimizer=tt.optim.SGD(lr=lr, momentum=momentum, weight_decay=0, nesterov=True))
        # tt.callbacks.EarlyStopping()
        # tt.callbacks.LRCosineAnnealing()
        callbacks = [tt.callbacks.EarlyStopping(), tt.callbacks.ClipGradNorm(model.net, 3.0)]
        verbose = True

        log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
            
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(X_test)
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        c_index = ev.concordance_td()
        # print(f"C-Index = {c_index:.4f}")

        C_indinces = []
        indices = np.arange(len(X_test))
        for _ in range(num_bootstrap):
            bootstraped_indices = np.random.choice(indices, size=len(X_test), replace=True)
            bootstrapped_X_test = X_test[bootstraped_indices]
            bootstrapped_y_test_T = durations_test[bootstraped_indices]
            bootstrapped_y_test_E = events_test[bootstraped_indices]
            
            surv = model.predict_surv_df(bootstrapped_X_test)
            ev = EvalSurv(surv, bootstrapped_y_test_T, bootstrapped_y_test_E, censor_surv='km')
            C_indinces.append(ev.concordance_td())
        
        c_index_lower = np.percentile(C_indinces, 2.5)
        c_index_upper = np.percentile(C_indinces, 97.5)
        print(f"C-Index (95% CI) ={c_index:.4f}({c_index_lower:.3f}, {c_index_upper:.3f})")

        torch.save(model.net, f'NSOTree_{dataset}.pth')
        # model.save_net()

        if vis:
            hr_pred = model.predict(viz_data['x'])
            hr_pred = np.array(hr_pred)
            with open(save_file, 'wb') as f:
                np.save(f, hr_pred)
    else:
        fold = 2

        ## hyperparameters that need to tuned commonly
        hidden_dim = 1
        depth = 14
        back_n = 1
        drop_type = 'node_dropconnect'
        p_dropout = 0.5
        net_type = 'locally_constant'
        anneal = 'approx'
        batch_size = 128
        epochs = 512
        lr = 0.2
        momentum = 0.9

        kfold_data = data
        fold_data = kfold_data[fold]
       
        train_data, valid_data = fold_data['train'], fold_data['val']
        train_data_df = format_dataset_to_df(train_data, 'T', 'E')
        X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

        val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
        X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

        get_target = lambda df: (df['T'].values, df['E'].values)
        y_train = get_target(train_data_df)
        y_valid = get_target(val_data_df)

        X_train = X_train.values.astype(np.float32)
        X_valid = X_valid.values.astype(np.float32)

        durations_val, events_val = get_target(val_data_df)
        val = (X_valid, y_valid)
            
        input_dim = X_train.shape[1]
        output_dim = 1
    
        model = CoxNOSTree(input_dim=input_dim, 
                                output_dim=output_dim, 
                                hidden_dim=hidden_dim, 
                                num_layer=depth, 
                                num_back_layer=back_n, 
                                dense=True, 
                                drop_type=drop_type, 
                                net_type=net_type, 
                                approx=anneal,
                                p_drop = p_dropout,
                                L1_reg = 1e-1,
                                optimizer=tt.optim.SGD(lr=lr, momentum=momentum, weight_decay=1e-4, nesterov=True))
        
        # model.optimizer.set_lr(lr)
        # epochs = 512
        callbacks = [tt.callbacks.EarlyStopping(), tt.callbacks.ClipGradNorm(model.net, 10.0)]
        verbose = True

        log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
            
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(X_valid)
        ev = EvalSurv(surv, durations_val, events_val, censor_surv='km')
        C_index = ev.concordance_td()
       
        print(f"[Fold={fold+1}] C-Index = {C_index:.4f}")