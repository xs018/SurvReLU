import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# For preprocessing
from sksurv.datasets import get_x_y
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

from utils.data_utils import load_dataset, format_dataset_to_df


if __name__ == "__main__":
    dataset = 'simulated_gaussian'
    data = load_dataset(dataset)

    random_state = 20
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)
    num_bootstrap = 100
    
    if 'simulated' in dataset:
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
        train_data_df = format_dataset_to_df(train_data, 'T', 'E')
        X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)
        val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
        X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)
        test_data_df = format_dataset_to_df(test_data, 'T', 'E')
        X_test, y_test = get_x_y(test_data_df, attr_labels=['E', 'T'], pos_label=1)

        ## label discretion
        num_durations = 10
        labtrans = DeepHitSingle.label_transform(num_durations)
        get_target = lambda df: (df['T'].values, df['E'].values)
        y_train = labtrans.fit_transform(*get_target(train_data_df))
        y_valid = labtrans.fit_transform(*get_target(val_data_df))
        y_test = labtrans.fit_transform(*get_target(test_data_df))

        train = (X_train.values, y_train)
        valid = (X_valid.values, y_valid)
        test = (X_test.values, y_test)

        # define neural net
        in_features = X_train.shape[1]
        num_nodes = [32, 32]
        out_features = labtrans.out_features
        batch_norm = True
        dropout = 0.1

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
        batch_size = 256
        lr_finder = model.lr_finder(train[0], train[1], batch_size, tolerance=3)
        _ = lr_finder.plot()

        model.optimizer.set_lr(0.01)

        epochs = 100
        callbacks = [tt.callbacks.EarlyStopping()]
        log = model.fit(train[0], train[1], batch_size, epochs, callbacks, val_data=test)
        _ = log.plot()

        C_indinces = []
        indices = np.arange(len(X_test))
        
        for _ in tqdm(range(num_bootstrap)):
            bootstraped_indices = np.random.choice(indices, size=len(X_test), replace=True)
            bootstrapped_X_test = X_test.iloc[bootstraped_indices].values
            bootstrapped_y_test_T = y_test[0][bootstraped_indices]
            bootstrapped_y_test_E = y_test[1][bootstraped_indices]
            
            surv = model.interpolate(10).predict_surv_df(bootstrapped_X_test)
            ev = EvalSurv(surv, bootstrapped_y_test_T, bootstrapped_y_test_E, censor_surv='km')
            C_indinces.append(ev.concordance_td('antolini'))
        
        c_index_lower = np.percentile(C_indinces, 2.5)
        c_index_upper = np.percentile(C_indinces, 97.5)

        surv = model.interpolate(10).predict_surv_df(test[0])
        ev = EvalSurv(surv, test[1][0], test[1][1], censor_surv='km')
        c_index = ev.concordance_td('antolini')

        print(f"C-Index (95% CI) ={c_index:.4f}({c_index_lower:.3f}, {c_index_upper:.3f})")

        # ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        # ev.concordance_td()

