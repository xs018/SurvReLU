import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

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
from pycox.models import CoxPH, CoxCC
from pycox.evaluation import EvalSurv

from utils.common_utils import get_logger, norm_hr
from utils.data_utils import load_dataset, format_dataset_to_df
from utils.vis_utils import plot_train_curve, plot_risk_model_2


if __name__ == "__main__":
    dataset = 'simulated_gaussian'
    data = load_dataset(dataset)

    random_state = 1234
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)
    num_bootstrap = 1

    if 'simulated' in dataset:
        vis = True
        save_file = f"{dataset}_Cox-MLP(Linear).npy"

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

        in_features = X_train.shape[1]
        num_nodes = []
        out_features = 1
        batch_norm = True
        dropout = 0.1
        output_bias = False

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                        dropout, output_bias=output_bias)
            
        model = CoxPH(net, tt.optim.Adam)
        batch_size = 512
        model.optimizer.set_lr(0.01)
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False

        log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
            
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(X_test)
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        c_index = ev.concordance_td()

        C_indinces = []
        indices = np.arange(len(X_test))
        
        for _ in tqdm(range(num_bootstrap)):
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

        if vis:
            # plot predicted risk surface
            # hr_pred = []
            # model.eval()
            # with torch.no_grad():
            #     for batch_idx, batch in enumerate(viz_dataloader):
            #         X = batch['X']
            #         logits = model(X)
            #         hr_pred.extend(logits.detach().cpu().numpy().squeeze())
            
            hr_pred = model.predict(viz_data['x'])
            hr_pred = np.array(hr_pred)
            # hr_pred = norm_hr(hr_pred)
            # hr_true = norm_hr(viz_data['hr'])
            # x_0 = viz_data_unnorm['x'][:, 0]
            # x_1 = viz_data_unnorm['x'][:, 1]
            # plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', save_file)
            with open(save_file, 'wb') as f:
                np.save(f, hr_pred)
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
            
            in_features = X_train.shape[1]
            num_nodes = [44, 44]
            out_features = 1
            batch_norm = True
            dropout = 0.2
            output_bias = False

            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                        dropout, output_bias=output_bias)
            
            model = CoxCC(net, tt.optim.Adam)
            batch_size = 2048
            model.optimizer.set_lr(0.02)
            epochs = 512
            callbacks = [tt.callbacks.EarlyStopping()]
            verbose = False

            log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
            
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(X_valid)
            ev = EvalSurv(surv, durations_val, events_val, censor_surv='km')
            C_index_folds.append(ev.concordance_td())
        
        c_index_mean = float(np.mean(C_index_folds))
        c_index_std = float(np.std(C_index_folds))
        print(f"C-Index = {c_index_mean:.4f} (+- {c_index_std:4f})")