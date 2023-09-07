import os
import numpy as np
import itertools

import torchtuples as tt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.common_utils import get_logger, norm_hr
from utils.args_utils import get_args
from utils.data_utils import load_cox_gaussian, SimpleDataset, FastTensorDataLoader
from models.relu_oblique_tree import ReLUObliqueTree
from losses import PartialLogLikelihood, calc_concordance_index, cox_ph_loss_sorted, CoxPHLoss, NLLLogistiHazardLoss, RankLoss, GANLoss


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


def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Generator(torch.nn.Module):
    def __init__(self, netH, netT):
        super().__init__()
        self.netH = netH
        self.netT = netT

    def forward(self, x):
        h = self.netH(x)
        t = self.netT(torch.exp(h))
        return h, t


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
    # model = ReLUObliqueTree(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type, net_type=args.net_type, approx=args.anneal).cuda()
    # print(model)
    num_nodes = [32, 32]
    netH = tt.practical.MLPVanilla(args.input_dim, num_nodes, args.output_dim, True,
                              0.1, output_bias=False).cuda()

    num_nodes_generator = [32, 32]
    generator_dropout = 0.5
    netT = tt.practical.MLPVanilla(args.output_dim, num_nodes, 1, True,
                              generator_dropout, output_bias=False).cuda()
    
    netG = Generator(netH, netT).cuda()
    
    num_nodes_discriminator = [16, 16]
    netD = tt.practical.MLPVanilla(2, num_nodes, 1, True,
                              0.0, output_bias=False).cuda()

    # define loss
    lambda_CoxPH = 100.0
    criterion_CoxPH = CoxPHLoss()
    
    lambda_L1 = 10.0
    criterion_L1 = torch.nn.L1Loss()

    # # Establish convention for real and fake labels during training
    # real_label = 1.
    # fake_label = 0.
    # Initialize the ``BCELoss`` function
    criterion_GAN = GANLoss('vanilla').cuda()

    # define optimizer and lr_scheduler
    if args.optimizer == 'SGD':
        # optimizer_H = optim.SGD(netH.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0)
        optimizer_G = optim.SGD(netG.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0)
        optimizer_D = optim.SGD(netD.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0)
    elif args.optimizer == 'AMSGrad':
        # optimizer_H = optim.Adam(netH.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
        optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
        optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
    elif args.optimizer == 'AdamW':
        # optimizer_H = optim.AdamW(netH.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
        optimizer_G = optim.AdamW(netG.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
        optimizer_D = optim.AdamW(netD.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.0)
    # scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
    best_score = -1e30
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch
    args.net_type = 'DeepSurv_Adv'

    best_model_name = './checkpoint/{}/{}/best_seed{}_ckpt.pth'.format(args.dataset.strip('/'), args.net_type, args.seed)
    last_model_name = './checkpoint/{}/{}/last_seed{}_ckpt.pth'.format(args.dataset.strip('/'), args.net_type, args.seed)
    
    model_dir = './checkpoint/{}/{}/'.format(args.dataset.strip('/'), args.net_type)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    log_file = './checkpoint/{}/{}/train.log'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    logger = get_logger(log_file)
    train_curve_file = './checkpoint/{}/{}/train_curve.jpg'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)

    best_hr_file = './checkpoint/{}/{}/best_seed{}_hr.jpg'.format(args.dataset.strip('/'), args.net_type, args.seed)
    last_hr_file = './checkpoint/{}/{}/last_seed{}_hr.jpg'.format(args.dataset.strip('/'), args.net_type, args.seed)

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
        netH.train()
        netG.train()
        netD.train()

        for batch_idx, batch in enumerate(train_dataloader):
            X = batch['X'].cuda()
            E = batch['E'].reshape(-1, ).cuda()
            T = batch['T'].reshape(-1, 1).cuda()

            ## forward
            h, fake_T = netG(X)

            # update D
            ## calculate GAN loss
            set_requires_grad(netD, True)
            # set_requires_grad([netH, netG], False)
            optimizer_D.zero_grad()
            # Fake; stop backprop to the generator by detaching fake_B
            fake_hT = torch.cat((h_exp.detach(), fake_T), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            
            pred_fake = netD(fake_hT)
            loss_D_fake = criterion_GAN(pred_fake, False)

            # Real
            real_hT = torch.cat((h_exp, T), 1)
            pred_real = netD(real_hT)
            loss_D_real = criterion_GAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # update H
            set_requires_grad(netD, False)
            optimizer_G.zero_grad()
            loss_CoxPH = criterion_CoxPH(h, T, E)
            loss_CoxPH.backward()
            optimizer_H.step()

            break

            # update G
            ## Calculate GAN and CoxPH Loss + L1 loss for the generator
            # set_requires_grad(netD, False)
            # optimizer_G.zero_grad()
            # # First, G(A) should fake the discriminator
            # fake_hT = torch.cat((h_exp, fake_T), 1)
            # pred_fake = netD(fake_hT.detach())
            # loss_G_GAN = criterion_GAN(pred_fake, True)
            # # Second, G(A) = B
            # loss_G_L1 = criterion_L1(fake_T, T) * lambda_L1
            # # Third, CoxPH Loss
            # loss_CoxPH = criterion_CoxPH(h, T, E) * lambda_CoxPH
            # # combine loss and calculate gradients
            # loss_G = loss_G_GAN + loss_G_L1 
            # loss_G.backward()
            # optimizer_G.step()

    #         train_loss += loss.item()

    #         pred_logits.extend(h.detach().cpu().numpy().squeeze())
    #         fail_indicator.extend(E.detach().cpu().numpy().squeeze())
    #         fail_time.extend(T.detach().cpu().numpy().squeeze())
        
    #     scheduler.step()

    #     pred_logits = np.array(pred_logits)
    #     fail_indicator = np.array(fail_indicator)
    #     fail_time = np.array(fail_time)
        
    #     train_loss = train_loss / len(train_dataloader)
    #     train_losses.append(train_loss)
    #     train_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
    #     train_CIs.append(train_CI)

    #     pred_logits = []
    #     fail_indicator = []
    #     fail_time = []
    #     val_loss = 0.0
    #     model.eval()
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(val_dataloader):
    #             X = batch['X']
    #             E = batch['E'].reshape(-1, )
    #             T = batch['T'].reshape(-1, )

    #             logits = model(X)
    #             loss = criterion(logits, T, E)

    #             val_loss += loss.item()

    #             pred_logits.extend(logits.detach().cpu().numpy().squeeze())
    #             fail_indicator.extend(E.detach().cpu().numpy().squeeze())
    #             fail_time.extend(T.detach().cpu().numpy().squeeze())

    #     pred_logits = np.array(pred_logits)
    #     fail_indicator = np.array(fail_indicator)
    #     fail_time = np.array(fail_time)
        
    #     val_loss = val_loss / len(val_dataloader)
    #     val_losses.append(val_loss)
    #     val_CI = calc_concordance_index(pred_logits, fail_indicator, fail_time)
    #     val_CIs.append(val_CI)
        
    #     logger.info(f"Epoch[{epoch}/{args.epochs}] train loss:{train_loss:.4f} train C-Index:{train_CI:.3f} val loss:{val_loss:.4f} val C-Index:{val_CI:.3f}")

    #     if val_CI > best_score:
    #         state = {
    #                 'model': model.state_dict(),
    #                 'metric': [train_losses, val_losses, train_CIs, val_CIs]
    #                  }
    #         torch.save(state, best_model_name)
    #         best_score = val_CI

    #     # "convergent" version
    #     state = {
    #              'model': model.state_dict(),
    #              'metric': [train_losses, val_losses, train_CIs, val_CIs]
    #              }
        
    #     torch.save(state, last_model_name)

    # ## plot training curve
    # plot_train_curve([train_losses, val_losses, train_CIs, val_CIs], train_curve_file)

    # # resume best weights
    # model.load_state_dict(torch.load(best_model_name)['model'])

    # # plot predicted risk surface
    # hr_pred = []
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(viz_dataloader):
    #         X = batch['X']
    #         logits = model(X)
    #         hr_pred.extend(logits.detach().cpu().numpy().squeeze())
            
    # hr_pred = np.array(hr_pred)
    # hr_pred = norm_hr(hr_pred)
    # hr_true = norm_hr(viz_data['hr'])
    # x_0 = viz_data_unnorm['x'][:, 0]
    # x_1 = viz_data_unnorm['x'][:, 1]
    # plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', best_hr_file)

    # # resume last weights
    # model.load_state_dict(torch.load(last_model_name)['model'])

    # # plot predicted risk surface
    # hr_pred = []
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(viz_dataloader):
    #         X = batch['X']
    #         logits = model(X)
    #         hr_pred.extend(logits.detach().cpu().numpy().squeeze())
            
    # hr_pred = np.array(hr_pred)
    # hr_pred = norm_hr(hr_pred)
    # hr_true = norm_hr(viz_data['hr'])
    # x_0 = viz_data_unnorm['x'][:, 0]
    # x_1 = viz_data_unnorm['x'][:, 1]
    # plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', last_hr_file)
    