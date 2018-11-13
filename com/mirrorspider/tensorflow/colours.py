import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

class TrueFalseCM():
    # f is 0, t is 1
    def __init__(self):
        f = np.array([0.75, 0.25, 0.25, 1])
        t = np.array([0.25, 0.75, 0.25, 1])
        tf = np.vstack((f, t))
        self.cm = ListedColormap(tf)
    def get_cmap(self):
        return self.cm
    
if __name__ == "__main__":
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    newcolors[:25, :] = pink
    newcmp = ListedColormap(newcolors)
    tfc = TrueFalseCM()
    newcmp = tfc.get_cmap()
    
    plot_examples([viridis, newcmp])