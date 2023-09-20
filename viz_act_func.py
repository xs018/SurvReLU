import numpy as np
import matplotlib.pyplot as plt


relu = lambda x: np.maximum(x, 0)

def my_softplus(x, tau=1., threshold=20.):
    truncate_mask = (x > threshold)
    return truncate_mask * x + (1. - truncate_mask) * (tau * np.log(1 + np.exp((1. - truncate_mask) * x / tau)))

softplus = lambda x: np.log(1 + np.exp(x)) * np.maximum(x, 20)

xx = np.linspace(-7, 7, 1000)
xv, yv = np.meshgrid(xx, xx)
z = np.logical_or(relu(xv), relu(yv))

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
# axes.plot(xx, relu(xx), c='C0')
# axes.plot(xx, my_softplus(xx), c='C1')
# axes.plot(xx, softplus(xx), c='C2')

# axes.imshow(relu(xv))
axes.contourf(xv, yv, z)
plt.axis('scaled')

# plt.savefig("")
plt.show()