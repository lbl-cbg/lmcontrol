import matplotlib.pyplot as plt

def imshow(img, ax=None, cmap='gray', vmin=0, vmax=255):
    if ax is None:
        ax = plt.gca()
    return ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
