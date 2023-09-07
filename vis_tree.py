import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt

base_dir = "checkpoint/sim_gaussian_2d/approx_locally_constant"
checkpoint = torch.load(f"{base_dir}/best_seed2_depth20_ckpt.pth")['model']
# print(checkpoint.keys())

xx1 = np.linspace(-1, 1, 2000)
print(xx1.shape)
for i in range(20):
    w = checkpoint[f'layer.{i}.weight'].cpu().numpy().squeeze()
    b = checkpoint[f'layer.{i}.bias'].cpu().numpy().squeeze()
    
    print(w)

    # w_norm1 = 

    xx2 = (-w[0] / w[1]) * xx1 - b / w[1]
    # a = 
    plt.plot(xx1, xx2)

    break

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()