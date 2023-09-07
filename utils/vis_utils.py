import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy
import numpy as np
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


def plot_train_curve(metric, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.plot(metric[0], color='b', ls='-', label="loss_train")
    ax.plot(metric[1], color='r', ls='-', label="loss_val")

    # ax2.plot(metric[2], color='g', ls='--', label="evaluation metric")
    ax2.plot(metric[3], color='g', ls='--', label="evaluation metric[C-Index]")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("evaluation metric")
    ax.legend()
    ax2.legend(loc=9)
    plt.title(f"[Epoch {len(metric[0])}] Best Metric={float(np.max(metric[3])):.4f}")
    # fig.savefig(join(output_folder, "progress.png"), dpi=300)
    fig.savefig(save_path, dpi=300)
    plt.close()


def plot_risk_model(x_0, x_1, hr, figsize=(4,3), clim = (-3,3), cmap = 'jet', save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.xlim(-1, 1)
    plt.xlabel('$x_0$', fontsize='large')
    plt.xticks(np.arange(-1, 1.5, .5))

    plt.ylim(-1, 1)
    plt.ylabel('$x_1$', fontsize='large')
    plt.yticks(np.arange(-1, 1.5, .5))
    
    im = plt.scatter(x=x_0, y=x_1, c=hr, marker='.', cmap=cmap)
    fig.colorbar(im)
    # plt.clim(0, 1)
    plt.clim(*clim)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()

    return (fig, ax, im)

def plot_risk_model_2(x_0, x_1, hr_true, hr_pred, figsize=(6,3), clim = (-3,3), cmap = 'jet', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.supxlabel('$x_0$', fontsize='large')
    fig.supylabel('$x_1$', fontsize='large')

    axes[0].set_xlim(-1, 1)
    
    axes[0].set_xticks(np.arange(-1, 1.5, .5))

    axes[0].set_ylim(-1, 1)
    axes[0].set_yticks(np.arange(-1, 1.5, .5))

    im = axes[0].scatter(x=x_0, y=x_1, c=hr_true, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    
    axes[1].set_xlim(-1, 1)
    # axes[1].set_xlabel('$x_0$', fontsize='large')
    axes[1].set_xticks(np.arange(-1, 1.5, .5))

    axes[1].set_ylim(-1, 1)
    # axes[1].set_ylabel('$x_1$', fontsize='large')
    axes[1].set_yticks([])

    im = axes[1].scatter(x=x_0, y=x_1, c=hr_pred, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)

    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cb_ax)
  

    plt.subplots_adjust(hspace=0.5, wspace=0.01)
    
    # plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()

