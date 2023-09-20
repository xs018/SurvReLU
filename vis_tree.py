import numpy as np
import torch
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt

from utils.common_utils import norm_hr
from utils.data_utils import load_dataset

dataset = 'simulated_gaussian_2d'
data = load_dataset(dataset)
viz_data = data['viz_unnorm']
hr_true = norm_hr(viz_data['hr'])
x_0 = viz_data['x'][:, 0]
x_1 = viz_data['x'][:, 1]

viz_data = data['viz']
x = torch.from_numpy(viz_data['x']).cuda()
print(x.size())

# clim = (-1, 1)
# im = ax.scatter(x=x_0, y=x_1, c=hr_true, marker='.', cmap='jet', vmin=0, vmax=1)
# im.set_clim(*clim)

net = torch.load(f"NSOTree_{dataset}.pth")
print(net)
checkpoint = net.state_dict()
net.eval()

with torch.no_grad():
    hr_pred = norm_hr(net(x).cpu().numpy().squeeze())

fig, axes = plt.subplots(2, net.num_layer + 1, figsize=((net.num_layer + 1) * 2, 4))
mask = np.ones_like(hr_pred).reshape(-1, )
pattern_aggr = np.ones_like(hr_pred).reshape(-1, )
for i in range(net.num_layer):
    relu_mask = net.relu_masks[i].cpu().numpy().reshape(-1, )
    pattern = net.patterns[i].cpu().numpy().reshape(-1, )
    im = axes[0, i].scatter(x=x_0, y=x_1, c=relu_mask, marker='.', cmap='gray', vmin=0, vmax=1)
    im = axes[1, i].scatter(x=x_0, y=x_1, c=pattern, marker='.', cmap='jet', vmin=0, vmax=1)
    
    mask = mask == relu_mask
    pattern_aggr += pattern
pattern_aggr /= pattern_aggr.max()

axes[0, -1].scatter(x=x_0, y=x_1, c=mask, marker='.', cmap='gray', vmin=0, vmax=1)
axes[1, -1].scatter(x=x_0, y=x_1, c=pattern_aggr, marker='.', cmap='jet', vmin=0, vmax=1)
plt.tight_layout()
plt.show()
