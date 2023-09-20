import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# from utils import print_args, AverageMeter
from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.common_utils import get_logger, norm_hr, AverageMeter
from utils.args_utils import get_args
from utils.data_utils import load_cox_gaussian, SimpleDataset, FastTensorDataLoader
from models.relu_oblique_tree import ReLUObliqueTree
from losses import PartialLogLikelihood, calc_concordance_index, cox_ph_loss_sorted, CoxPHLoss, NLLLogistiHazardLoss, RankLoss
from utils.data_utils import load_dataset


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


def train(args, model, train_loader, optimizer, criterion, anneal, train_pred, offset, feat_indices, data_indices, alpha=1):
    model.train()
    avg_loss = AverageMeter()

    train_x = train_loader.tensors[0][:, feat_indices]
    train_target_T = train_loader.tensors[2]
    train_target_E = train_loader.tensors[1]

    np.random.shuffle(data_indices)

    for data_idx in range(0, len(data_indices), args.batch_size):
        data = train_x[data_indices[data_idx:data_idx+args.batch_size]].clone().detach()
        target_T = train_target_T[data_indices[data_idx:data_idx+args.batch_size]].clone().detach()
        target_E = train_target_E[data_indices[data_idx:data_idx+args.batch_size]].clone().detach()

        base = np.sum(train_pred[data_indices[data_idx:data_idx+args.batch_size]], axis=1) + offset
        base = torch.tensor(base).type(torch.cuda.FloatTensor)
                
        optimizer.zero_grad()
        ###############
        data.requires_grad = True
        if args.p != -1:
            assert(args.p >= 0. and args.p < 1)
            logits, relu_masks = model(data, alpha=alpha, anneal=anneal, p=args.p, training=True)
        else:
            logits, relu_masks = model(data, alpha=alpha, anneal=anneal, p=1-alpha, training=True)
        ###############

        logits = logits + base

        loss = criterion(logits, target_T, target_E)

        L1_reg = 0.0
        for name, param in model.layer.named_parameters():
            if 'weight' in name:
                L1_reg += param.data.abs().sum()

        loss = loss + args.reg * L1_reg
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        avg_loss.update(loss.item())
        
    return avg_loss.avg


def predict(model, device, test_loader, feat_indices):
    model.eval()
    score = []
    for batch in test_loader:
        data = batch['X']
        data = data[:, feat_indices]
        # target_E = batch['E'].reshape(-1, )
        # target_T = batch['T'].reshape(-1, )

        # data, data.to(device), target_E.to(device), target_T.to(device)
    
        ###############
        data.requires_grad = True
        if model.net_type == 'locally_constant':
            logits, relu_masks = model(data)
        elif model.net_type == 'locally_linear':
            logits, relu_masks = model.normal_forward(data)
        ###############

        score += list(logits.cpu().data.numpy())

    return np.array(score)


if __name__ == "__main__":
    fold = 0
    # get hyperparameters
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set global seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kfold_data = load_dataset(args.dataset)
    fold_data = kfold_data[fold]
    train_data, valid_data = fold_data['train'], fold_data['val']
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

    train_target_E, train_target_T = train_dataloader.tensors[1].clone().cpu().numpy(), train_dataloader.tensors[2].clone().cpu().numpy()
    val_target_E, val_target_T = val_dataloader.tensors[1].clone().cpu().numpy(), val_dataloader.tensors[2].clone().cpu().numpy()

    args.input_dim = train_data['x'].shape[-1]
    args.output_dim = 1
    
    train_pred = np.zeros((train_dataloader.dataset_len, args.ensemble_n, args.output_dim), dtype=np.float32)
    valid_pred = np.zeros((val_dataloader.dataset_len, args.ensemble_n, args.output_dim), dtype=np.float32)

    offset = np.array([[0.]], dtype=np.float32)

    for ensemble_idx in range(args.ensemble_n):
        feat_indices = np.arange(args.input_dim)
        feat_dim = args.input_dim
        data_indices = np.arange(train_dataloader.dataset_len)
        
        model = ReLUObliqueTree(input_dim=args.input_dim, 
                                output_dim=args.output_dim, 
                                hidden_dim=args.hidden_dim, 
                                num_layer=args.depth, 
                                num_back_layer=args.back_n, 
                                dense=True, 
                                drop_type=args.drop_type, 
                                net_type=args.net_type, 
                                approx=args.anneal).cuda()

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

        if ensemble_idx == 0:
            shrinkage = 1
        else:
            shrinkage = args.shrinkage

        for epoch in range(start_epoch, args.epochs + start_epoch):
            alpha = get_alpha(epoch, args.epochs)
            
            train_approximate_loss = train(args, model, train_dataloader, optimizer, criterion, args.anneal, train_pred, offset, feat_indices, data_indices, alpha)
            scheduler.step(epoch)

        train_pred[:, ensemble_idx, :] = shrinkage * predict(model, device, train_dataloader, feat_indices)
        valid_pred[:, ensemble_idx, :] = shrinkage * predict(model, device, val_dataloader, feat_indices)

        # print(np.isnan(valid_pred).any())
        # print(np.isnan(val_target_E).any())
        # print(np.isnan(val_target_T).any())
        train_c_index = calc_concordance_index(np.sum(train_pred, axis=1).flatten() + offset[0], train_target_E, train_target_T)
        val_c_index = calc_concordance_index(np.sum(valid_pred, axis=1).flatten() + offset[0], val_target_E, val_target_T)

        print(f"Iteration {ensemble_idx+1}  C-index: [{train_c_index:.4f} / {val_c_index:.4f}] ")

    # C_index_folds = []
    # for fold_idx, fold_data in enumerate(kfold_data):
    #     train_data, valid_data = fold_data['train'], fold_data['val']
    #     train_dataloader =  FastTensorDataLoader(torch.from_numpy(train_data['x']).float().cuda(), 
    #                         torch.from_numpy(train_data['e']).float().cuda(),
    #                         torch.from_numpy(train_data['t']).float().cuda(),
    #                         tensor_names=('X', 'E', 'T'),
    #                         batch_size=args.batch_size, shuffle=True)
            
    #     val_dataloader =  FastTensorDataLoader(torch.from_numpy(valid_data['x']).float().cuda(), 
    #                         torch.from_numpy(valid_data['e']).float().cuda(),
    #                         torch.from_numpy(valid_data['t']).float().cuda(),
    #                         tensor_names=('X', 'E', 'T'),
    #                         batch_size=args.batch_size, shuffle=False)

    #     args.input_dim = train_data['x'].shape[-1]
    #     args.output_dim = 1

    #     # define network 
    #     model = ReLUObliqueTree(input_dim=args.input_dim, 
    #                             output_dim=args.output_dim, 
    #                             hidden_dim=args.hidden_dim, 
    #                             num_layer=args.depth, 
    #                             num_back_layer=args.back_n, 
    #                             dense=True, 
    #                             drop_type=args.drop_type, 
    #                             net_type=args.net_type, 
    #                             approx=args.anneal).cuda()

    #     # define loss
    #     criterion = CoxPHLoss()

    #     # define optimizer and lr_scheduler
    #     if args.optimizer == 'SGD':
    #         optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0)
    #     elif args.optimizer == 'AMSGrad':
    #         optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
    #     elif args.optimizer == 'AdamW':
    #         optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
    #     scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
        
    #     best_score = -1e30
    #     start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    #     train_losses = []
    #     val_losses = []
    #     train_CIs = []
    #     val_CIs = []
    #     for epoch in range(start_epoch, args.epochs + start_epoch):
    #         alpha = get_alpha(epoch, args.epochs)

    #         pred_logits = []
    #         fail_indicator = []
    #         fail_time = []
    #         train_loss = 0.0
    #         model.train()
    #         for batch_idx, batch in enumerate(train_dataloader):
    #             X = batch['X']
    #             E = batch['E'].reshape(-1, )
    #             T = batch['T'].reshape(-1, )

    #             logits, relu_masks = model(X)
    #             loss = criterion(logits, T, E)

    #             L1_reg = 0.0
    #             for name, param in model.layer.named_parameters():
    #                 if 'weight' in name:
    #                     L1_reg += param.data.abs().sum()

    #             loss = loss + args.reg * L1_reg

    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()

    #             # model.layer.parameters() = softThresholding(model.layer.parameters(), 0.1)

    #             train_loss += loss.item()

    #             pred_logits.extend(logits.detach().cpu().numpy().squeeze())
    #             fail_indicator.extend(E.detach().cpu().numpy().squeeze())
    #             fail_time.extend(T.detach().cpu().numpy().squeeze())
            
    #         scheduler.step()

    #         pred_logits = np.array(pred_logits)
    #         fail_indicator = np.array(fail_indicator)
    #         fail_time = np.array(fail_time)
            
    #         train_loss = train_loss / len(train_dataloader)
    #         train_losses.append(train_loss)
    #         train_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
    #         train_CIs.append(train_CI)

    #         pred_logits = []
    #         fail_indicator = []
    #         fail_time = []
    #         val_loss = 0.0
    #         model.eval()
    #         with torch.no_grad():
    #             for batch_idx, batch in enumerate(val_dataloader):
    #                 X = batch['X']
    #                 E = batch['E'].reshape(-1, )
    #                 T = batch['T'].reshape(-1, )

    #                 logits, relu_masks = model(X)
    #                 loss = criterion(logits, T, E)

    #                 val_loss += loss.item()

    #                 pred_logits.extend(logits.detach().cpu().numpy().squeeze())
    #                 fail_indicator.extend(E.detach().cpu().numpy().squeeze())
    #                 fail_time.extend(T.detach().cpu().numpy().squeeze())

    #         pred_logits = np.array(pred_logits)
    #         fail_indicator = np.array(fail_indicator)
    #         fail_time = np.array(fail_time)
            
    #         val_loss = val_loss / len(val_dataloader)
    #         val_losses.append(val_loss)
    #         val_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
    #         val_CIs.append(val_CI)

    #         if val_CI > best_score:
    #             best_score = val_CI
            
    #         # logger.info(f"Epoch[{epoch}/{args.epochs}] train loss:{train_loss:.4f} train C-Index:{train_CI:.3f} val loss:{val_loss:.4f} val C-Index:{val_CI:.3f}")

    #     C_index_folds.append(best_score)

    # C_index_mean = float(np.mean(C_index_folds))
    # C_index_std = float(np.std(C_index_folds))

    # print(f"C-Index = {C_index_mean:.4f} (+- {C_index_std:4f})")

    