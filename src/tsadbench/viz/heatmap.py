"""Shared annotated-heatmap helpers (from Matplotlib's example gallery).

Single source for the sensitivity-plot tools so the helpers are not copied across
the sensitivity figure scripts in tools/.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """Create a heatmap from a numpy array and two lists of labels.
    Taken from the Matplotlib example gallery (minor edits: defaults)."""
    if cbar_kw is None:
        cbar_kw = {}

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Turn spines off and create white grid
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """Annotate a heatmap.
    Taken from the Matplotlib example gallery (minor edits: NaN skip)."""
    if not isinstance(data, (list, np.ndarray, np.ma.MaskedArray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        # If no threshold, use middle of colormap
        if isinstance(im.norm, Normalize) and (im.norm.vmin is not None and im.norm.vmax is not None):
            threshold = im.norm((im.norm.vmin + im.norm.vmax) / 2.0)

    # Default alignment
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Choose colormap-dependent color for value
    if isinstance(valfmt, str):
        valfmt = plt.matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Skip masked/NaN cells
            if np.ma.is_masked(val) or (isinstance(val, float) and np.isnan(val)):
                continue
            color_index = im.norm(val)
            kw.update(color=textcolors[int(color_index > (threshold if threshold is not None else 0.5))])
            text = im.axes.text(j, i, valfmt(val, None), **kw)
            texts.append(text)

    return texts
