import matplotlib.pyplot as plt
import numpy as np


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, 
           edgecolors=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, 
                   linewidths=linewidths, verts=verts, edgecolors=None, **kwargs)
    
if __name__ == "__main__":
    x = np.random.randint(1, 5, 100)
    y = np.random.randint(1, 5, 100)
    plt.figure()
    jitter(x, y, c='r', marker='>')
    plt.show()
    