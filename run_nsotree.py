import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.common_utils import get_logger, norm_hr
from utils.args_utils import get_args
from utils.data_utils import load_dataset, SimpleDataset, FastTensorDataLoader
from models.relu_oblique_tree import ReLUObliqueTree
from losses import PartialLogLikelihood, calc_concordance_index, cox_ph_loss_sorted, CoxPHLoss, NLLLogistiHazardLoss, RankLoss


def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)

def fit(args, model, train_dataloader, test_dataloader):
    # define loss
    criterion = CoxPHLoss()

    # define optimizer and lr_scheduler
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0)
    elif args.optimizer == 'AMSGrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
    best_score = -1e30
    best_weight = None
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    for epoch in range(start_epoch, args.epochs + start_epoch):
        alpha = get_alpha(epoch, args.epochs)

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            X = batch['X']
            E = batch['E'].reshape(-1, )
            T = batch['T'].reshape(-1, )

            logits, relu_masks = model(X)
            loss = criterion(logits, T, E)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pred_logits = []
        fail_indicator = []
        fail_time = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                X = batch['X']
                E = batch['E'].reshape(-1, )
                T = batch['T'].reshape(-1, )

                logits, relu_masks = model(X)
                loss = criterion(logits, T, E)

                pred_logits.extend(logits.detach().cpu().numpy().squeeze())
                fail_indicator.extend(E.detach().cpu().numpy().squeeze())
                fail_time.extend(T.detach().cpu().numpy().squeeze())

        pred_logits = np.array(pred_logits)
        fail_indicator = np.array(fail_indicator)
        fail_time = np.array(fail_time)

        val_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
        print(f"=== Epoch [{epoch}/{args.epochs}] C-Index:{val_CI:.4f} ===")
        if val_CI > best_score:
            best_score = val_CI
            torch.save(model.state_dict(), "best_model.pt")

        scheduler.step()

    print(f"Best C-Index={best_score:.4f}")


def predict(model, test_data, num_bootstrap=100):
    model.load_state_dict(torch.load("best_model.pt"))

    X_test = test_data['x']
    T_test = test_data['t']
    E_test = test_data['e']

    C_indinces = []
    indices = np.arange(len(X_test))

    for _ in tqdm(range(num_bootstrap)):
        bootstraped_indices = np.random.choice(indices, size=len(X_test), replace=True)
        bootstrapped_X_test = X_test[bootstraped_indices] 
        bootstrapped_T_test = T_test[bootstraped_indices] 
        bootstrapped_E_test = E_test[bootstraped_indices] 

        model.eval()
        with torch.no_grad():
            bootstrapped_X_test_torch = torch.from_numpy(bootstrapped_X_test).cuda()
            logits, relu_masks = model(bootstrapped_X_test_torch)
            logits_numpy = logits.detach().cpu().numpy().squeeze()

            val_CI = calc_concordance_index(logits_numpy, bootstrapped_E_test, bootstrapped_T_test)
            C_indinces.append(val_CI)
    
    c_index_lower = np.percentile(C_indinces, 2.5)
    c_index_upper = np.percentile(C_indinces, 97.5)
    
    model.eval()
    with torch.no_grad():
        X_test_torch = torch.from_numpy(X_test).cuda()
        logits, relu_masks = model(X_test_torch)
        logits_numpy = logits.detach().cpu().numpy().squeeze()
    
    c_index = calc_concordance_index(logits_numpy, E_test, T_test)
    print(f"C-Index (95% CI) ={c_index:.4f}({c_index_lower:.4f}, {c_index_upper:.4f})")

if __name__ == "__main__":
    dataset = 'simulated_gaussian'
    data = load_dataset(dataset)

    # get hyperparameters
    args = get_args()

    # set global seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_bootstrap = 100

    if 'simulated' in dataset:
        args.input_dim = 10
        args.output_dim = 1

        train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']

        train_dataloader =  FastTensorDataLoader(torch.from_numpy(train_data['x']).float().cuda(), 
                        torch.from_numpy(train_data['e']).float().cuda(),
                        torch.from_numpy(train_data['t']).float().cuda(),
                        tensor_names=('X', 'E', 'T'),
                        batch_size=args.batch_size, shuffle=True)
        
        val_dataloader =  FastTensorDataLoader(torch.from_numpy(valid_data['x']).float().cuda(), 
                        torch.from_numpy(valid_data['e']).float().cuda(),
                        torch.from_numpy(valid_data['t']).float().cuda(),
                        tensor_names=('X', 'E', 'T'),
                        batch_size=args.batch_size, shuffle=False)

        test_dataloader =  FastTensorDataLoader(torch.from_numpy(test_data['x']).float().cuda(), 
                        torch.from_numpy(test_data['e']).float().cuda(),
                        torch.from_numpy(test_data['t']).float().cuda(),
                        tensor_names=('X', 'E', 'T'),
                        batch_size=args.batch_size, shuffle=False)

    # define network 
    model = ReLUObliqueTree(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type, net_type=args.net_type, approx=args.anneal).cuda()
    
    fit(args, model, train_dataloader, test_dataloader)
    predict(model, test_data, num_bootstrap)