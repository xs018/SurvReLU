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


if __name__ == "__main__":
    # get hyperparameters
    fold = 3
    args = get_args()

    # set global seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kfold_data = load_dataset(args.dataset)
    C_index_folds = []
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

    args.input_dim = train_data['x'].shape[-1]
    args.output_dim = 1

        # define network
    model = ReLUObliqueTree(input_dim=args.input_dim,
                            output_dim=args.output_dim,
                            hidden_dim=args.hidden_dim,
                            num_layer=args.depth,
                            num_back_layer=args.back_n,
                            dense=True,
                            drop_type=args.drop_type,
                            net_type=args.net_type,
                            approx=args.anneal,
                            reg = args.reg).cuda()

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

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # for idx, (name, param) in enumerate(model.layer.named_parameters()):
                #     if 'weight' in name and idx > 1:
                #         # param.data = param.data / param.data.sum()
                #         param.data = softThresholding(param.data, args.reg)
                        # param.data = param.data / param.data.sum()

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
            relu_masks_all = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    X = batch['X']
                    E = batch['E'].reshape(-1, )
                    T = batch['T'].reshape(-1, )

                    logits, relu_masks = model(X)
                    relu_masks_all += relu_masks
                    loss = criterion(logits, T, E)

                    val_loss += loss.item()

                    pred_logits.extend(logits.detach().cpu().numpy().squeeze())
                    fail_indicator.extend(E.detach().cpu().numpy().squeeze())
                    fail_time.extend(T.detach().cpu().numpy().squeeze())
            print(len(relu_masks_all))
            pred_logits = np.array(pred_logits)
            fail_indicator = np.array(fail_indicator)
            fail_time = np.array(fail_time)

            val_loss = val_loss / len(val_dataloader)
            val_losses.append(val_loss)
            val_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
            val_CIs.append(val_CI)

            if val_CI > best_score:
                save_this = {
                     'state': model.state_dict(),
                     'relu_masks': relu_masks_all,
                     'pred_hr': pred_logits,
                     'valid_data': valid_data,
                     'model': model
                }
                torch.save(save_this, f"viz/best_{args.dataset}.pth")

                best_score = val_CI

        #     break
                print(best_score)
            # logger.info(f"Epoch[{epoch}/{args.epochs}] train loss:{train_loss:.4f} train C-Index:{train_CI:.3f} val loss:{val_loss:.4f} val C-Index:{val_CI:.3f}")

    # C_index_folds.append(best_score)
    # print(C_index_folds)
    # C_index_mean = float(np.mean(C_index_folds))
    # C_index_std = float(np.std(C_index_folds))

    # print(f"C-Index = {C_index_mean:.4f} (+- {C_index_std:4f})")

