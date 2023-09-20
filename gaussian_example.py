import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sksurv.datasets import get_x_y

import torch
import torchtuples as tt
from torchtuples import Model, optim

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from models.nostree import CoxNOSTree, CoxCCNOSTree
from models.notree import NOTree

from utils.common_utils import norm_hr
from utils.data_utils import load_dataset, format_dataset_to_df


if __name__ == "__main__":
    # dataset = 'simulated_linear_2d'
    # data = load_dataset(dataset)

    # random_state = 1234
    # np.random.seed(random_state)
    # _ = torch.manual_seed(random_state)

    # hidden_dim = 1
    # depth = 2
    # back_n =  0
    # drop_type = 'node_dropconnect'
    # p_dropout = 0.2
    # net_type = 'locally_constant'
    # anneal = 'approx'
    # batch_size = 256
    # epochs = 512
    # lr = 0.1
    # momentum = 0.95

    # train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
    # net = NOTree(2, 
    #             1, 
    #             hidden_dim, 
    #             depth, 
    #             back_n, 
    #             dense = True, 
    #             drop_type = drop_type, 
    #             net_type = net_type, 
    #             approx = anneal)
    
    # loss = torch.nn.MSELoss()
    # model = tt.Model(net, loss, tt.optim.SGD(lr=lr, momentum=momentum, nesterov=True))
    
    # callbacks = [tt.callbacks.EarlyStopping(), tt.callbacks.LRCosineAnnealing()]
    # verbose = True
    # # model.fit(train_data['x'], norm_hr(train_data['hr']).reshape(-1, 1), batch_size, epochs, callbacks, verbose, val_data=(valid_data['x'], norm_hr(valid_data['hr']).reshape(-1, 1)), val_batch_size=batch_size)

    # data = torch.from_numpy(viz_data['x']).cuda()
    # data.requires_grad = True

    # model.net.eval()

    # print(model.net)
    # y_pred = model.net(data, p=0).detach().cpu().numpy().squeeze()

    # x_0 = viz_data_unnorm['x'][:, 0]
    # x_1 = viz_data_unnorm['x'][:, 1]

    # with open("viz/viz_network.pkl", 'wb') as f:
    #     pickle.dump({
    #         'x0': x_0,
    #         'x1': x_1,
    #         'hr_true': norm_hr(viz_data_unnorm['hr']),
    #         'hr_pred': y_pred,
    #         'relu_masks': net.relu_masks,
    #         'patterns': net.patterns,
    #     }, f)

    with open("viz/viz_network.pkl", 'rb') as f:
        cache = pickle.load(f)

    x_0, x_1 = cache['x0'], cache['x1']
    hr_true, hr_pred = cache['hr_true'], cache['hr_pred']
    relu_masks, patterns = cache['relu_masks'], cache['patterns']

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    clim = (-1, 1)
    im = ax[0].scatter(x=x_0, y=x_1, c=hr_true, marker='.', cmap='jet', vmin=0, vmax=1)
    im.set_clim(*clim)
    fig.colorbar(im, ax=ax[0])
    # ax[0].set_title("$y_{true}$")
    clim = (-1, 1)
    im = ax[1].scatter(x=x_0, y=x_1, c=hr_pred, marker='.', cmap='jet', vmin=0, vmax=1)
    im.set_clim(*clim)
    fig.colorbar(im, ax=ax[1])
    # ax[1].set_title("$h(\boldsymbol{x}) = x_0 + 2 x_1$")
    plt.tight_layout()
    plt.savefig("viz/hr.jpg", dpi=300)
    plt.close('all')

    num_layer = len(relu_masks)

    fig, axes = plt.subplots(num_layer+2, 2, figsize=(6, (num_layer+2) * 2 ))
    # mask = np.ones_like(hr_true).reshape(-1, )
    # pattern_aggr = np.zeros_like(hr_true).reshape(-1, )
    # for i in range(num_layer):
    #     relu_mask = relu_masks[i].detach().cpu().numpy().reshape(-1, )
    #     pattern = 1- patterns[i].detach().cpu().numpy().reshape(-1, )
    #     im = axes[i, 0].scatter(x=x_0, y=x_1, c=relu_mask, marker='.', cmap='gray', vmin=0, vmax=1)
    #     im.set_clim(*(0, 1))
    #     im = axes[i, 1].scatter(x=x_0, y=x_1, c=pattern, marker='.', cmap='jet', vmin=0, vmax=1)
    #     im.set_clim(*(-1, 1))

    #     mask = mask == relu_mask
    #     pattern_aggr += pattern
        
    # xx = np.linspace(-1, 1, 1000)
    # axes[0, 0].plot(xx, -0.5 * xx - 0.5, c='red', linewidth=2)
    # axes[1, 0].plot(xx, -0.5 * xx + 0.5, c='blue', linewidth=2)
    # axes[2, 0].plot(xx, -0.5 * xx - 0.5, c='red', linewidth=2)
    # axes[2, 0].plot(xx, -0.5 * xx + 0.5, c='blue', linewidth=2)

    # mask = mask / mask.max()
    # pattern_aggr = pattern_aggr / num_layer

    # im = axes[-2, 0].scatter(x=x_0, y=x_1, c=mask, marker='.', cmap='gray', vmin=0, vmax=1)
    # im.set_clim(*(0, 1))
    # im = axes[-2, 1].scatter(x=x_0, y=x_1, c=mask, marker='.', cmap='jet', vmin=0, vmax=1)
    # im.set_clim(*(-1, 1))
    
    axes[-1, 0].set_axis_off()
    axes[-1, 1].set_axis_off()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    cax = axes[-2, 1].inset_axes([1.04, 0.04 , 0.06, 0.9])
    fig.colorbar(im, ax=axes[-2, 1], cax=cax)
    legend_color = ['white', 'black', 'red', 'blue']
    patch1 = mpatches.Patch(color=legend_color[0], label='non-activated')
    patch2 = mpatches.Patch(color=legend_color[1], label='activated')
    patch3 = mpatches.Patch(color=legend_color[2], label="$x_0 + 2 x_1 + 1 = 0$")
    patch4 = mpatches.Patch(color=legend_color[3], label="$x_0 + 2 x_1 - 1 = 0$")
    legend = fig.legend(handles=[patch1, patch2, patch3, patch4], loc="upper left", ncol=2, fontsize=8)
    frame = legend.get_frame()
    frame.set_facecolor('silver')
    plt.savefig("viz/symbol.jpg", dpi=500)