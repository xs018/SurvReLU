import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.common_utils import get_logger, norm_hr
from utils.args_utils import get_args
from utils.data_utils import load_cox_gaussian, SimpleDataset, FastTensorDataLoader
from models.relu_oblique_tree import ReLUObliqueTree
from losses import PartialLogLikelihood, calc_concordance_index, cox_ph_loss_sorted, CoxPHLoss, NLLLogistiHazardLoss, RankLoss


def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)

def prox_plus(w):
        """Projection onto non-negative numbers
        """
        below = w < 0
        w[below] = 0
        return w

def softThresholding(w, lam=0.1):
    return torch.sign(w) * prox_plus(torch.abs(w) - lam)


if __name__ == "__main__":
    # get hyperparameters
    args = get_args()

    # set global seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'sim_gaussian':
        args.input_dim = 10
        args.output_dim = 1

        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_cox_gaussian()

        # print(valid_data['hr'])

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

        viz_dataloader = FastTensorDataLoader(torch.from_numpy(viz_data['x']).float().cuda(), 
                        torch.from_numpy(viz_data['e']).float().cuda(),
                        torch.from_numpy(viz_data['t']).float().cuda(),
                        tensor_names=('X', 'E', 'T'),
                        batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'sim_gaussian_2d':
        args.input_dim = 2
        args.output_dim = 1

        train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_cox_gaussian("datasets/gaussian_survival_data_2d.h5")

        # print(valid_data['hr'])

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

        viz_dataloader = FastTensorDataLoader(torch.from_numpy(viz_data['x']).float().cuda(), 
                        torch.from_numpy(viz_data['e']).float().cuda(),
                        torch.from_numpy(viz_data['t']).float().cuda(),
                        tensor_names=('X', 'E', 'T'),
                        batch_size=args.batch_size, shuffle=False)
    else:
        raise Exception("not supported yet !")

    # define network 
    model = ReLUObliqueTree(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type, net_type=args.net_type, approx=args.anneal).cuda()
    print(model)

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
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch
    if args.anneal == 'approx':
        args.net_type = 'approx_' + args.net_type 

    best_model_name = './checkpoint/{}/{}/best_seed{}_depth{}_ckpt.pth'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    last_model_name = './checkpoint/{}/{}/last_seed{}_depth{}_ckpt.pth'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    
    model_dir = './checkpoint/{}/{}/'.format(args.dataset.strip('/'), args.net_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    log_file = './checkpoint/{}/{}/train.log'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    logger = get_logger(log_file)
    train_curve_file = './checkpoint/{}/{}/train_curve.jpg'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)

    best_hr_file = './checkpoint/{}/{}/best_seed{}_depth{}_hr.jpg'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    last_hr_file = './checkpoint/{}/{}/last_seed{}_depth{}_hr.jpg'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)

    train_losses = []
    val_losses = []
    train_CIs = []
    val_CIs = []
    for epoch in range(start_epoch, args.epochs + start_epoch):
        alpha = get_alpha(epoch, args.epochs)

        pred_logits = []
        fail_indicator = []
        fail_time = []
        train_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            X = batch['X']
            E = batch['E'].reshape(-1, )
            T = batch['T'].reshape(-1, )

            logits, relu_masks = model(X)
            loss = criterion(logits, T, E)

            L1_reg = 0.0
            for name, param in model.layer.named_parameters():
                if 'weight' in name:
                    L1_reg += param.data.abs().sum()

            loss = loss + args.reg * L1_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # model.layer.parameters() = softThresholding(model.layer.parameters(), 0.1)

            train_loss += loss.item()

            pred_logits.extend(logits.detach().cpu().numpy().squeeze())
            fail_indicator.extend(E.detach().cpu().numpy().squeeze())
            fail_time.extend(T.detach().cpu().numpy().squeeze())
        
        scheduler.step()

        pred_logits = np.array(pred_logits)
        fail_indicator = np.array(fail_indicator)
        fail_time = np.array(fail_time)
        
        train_loss = train_loss / len(train_dataloader)
        train_losses.append(train_loss)
        train_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
        train_CIs.append(train_CI)

        pred_logits = []
        fail_indicator = []
        fail_time = []
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                X = batch['X']
                E = batch['E'].reshape(-1, )
                T = batch['T'].reshape(-1, )

                logits, relu_masks = model(X)
                loss = criterion(logits, T, E)

                val_loss += loss.item()

                pred_logits.extend(logits.detach().cpu().numpy().squeeze())
                fail_indicator.extend(E.detach().cpu().numpy().squeeze())
                fail_time.extend(T.detach().cpu().numpy().squeeze())

        pred_logits = np.array(pred_logits)
        fail_indicator = np.array(fail_indicator)
        fail_time = np.array(fail_time)
        
        val_loss = val_loss / len(val_dataloader)
        val_losses.append(val_loss)
        val_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
        val_CIs.append(val_CI)
        
        logger.info(f"Epoch[{epoch}/{args.epochs}] train loss:{train_loss:.4f} train C-Index:{train_CI:.3f} val loss:{val_loss:.4f} val C-Index:{val_CI:.3f}")

        if val_CI > best_score:
            state = {
                    'model': model.state_dict(),
                    'metric': [train_losses, val_losses, train_CIs, val_CIs]
                     }
            torch.save(state, best_model_name)
            best_score = val_CI

        # "convergent" version
        state = {
                 'model': model.state_dict(),
                 'metric': [train_losses, val_losses, train_CIs, val_CIs]
                 }
        
        torch.save(state, last_model_name)

    ## plot training curve
    plot_train_curve([train_losses, val_losses, train_CIs, val_CIs], train_curve_file)

    # resume best weights
    model.load_state_dict(torch.load(best_model_name)['model'])

    # plot predicted risk surface
    hr_pred = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(viz_dataloader):
            X = batch['X']
            logits, relu_masks = model(X)
            hr_pred.extend(logits.detach().cpu().numpy().squeeze())
            
    hr_pred = np.array(hr_pred)
    hr_pred = norm_hr(hr_pred)
    hr_true = norm_hr(viz_data['hr'])
    x_0 = viz_data_unnorm['x'][:, 0]
    x_1 = viz_data_unnorm['x'][:, 1]
    plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', best_hr_file)

    # resume last weights
    model.load_state_dict(torch.load(last_model_name)['model'])

    # plot predicted risk surface
    hr_pred = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(viz_dataloader):
            X = batch['X']
            logits, relu_masks = model(X)
            hr_pred.extend(logits.detach().cpu().numpy().squeeze())
            
    hr_pred = np.array(hr_pred)
    hr_pred = norm_hr(hr_pred)
    hr_true = norm_hr(viz_data['hr'])
    x_0 = viz_data_unnorm['x'][:, 0]
    x_1 = viz_data_unnorm['x'][:, 1]
    plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', last_hr_file)
    