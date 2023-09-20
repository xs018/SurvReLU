import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

from tqdm import tqdm
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

def depth_ablation(depth):
    dataset = 'simulated_gaussian'
    data = load_dataset(dataset)

    random_state = 1234
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)

    hidden_dim = 1
    back_n =  0
    drop_type = 'node_dropconnect'
    p_dropout = 0.5
    net_type = 'locally_constant'
    anneal = 'approx'
    batch_size = 512
    epochs = 512
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
                                L1_reg = 1e-4,
                                optimizer=tt.optim.SGD(lr=lr, momentum=momentum, weight_decay=0, nesterov=True))
        
    callbacks = [tt.callbacks.EarlyStopping(), tt.callbacks.ClipGradNorm(model.net, 10), tt.callbacks.LRCosineAnnealing()]
    verbose = False

    log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
            
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(X_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td()

    print(f"depth={depth} C-index:{c_index:.4f}")

    return c_index

if __name__ == "__main__":
   depths = [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24]
#    depths =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 28, 32, 36, 40]
   c_indexs = []
   for d in tqdm(depths):
       c_indexs.append(depth_ablation(d))

   with open('viz/depth_ablation.npy', 'wb') as f:
       np.save(f, np.array(c_indexs))
