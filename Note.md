I modified the viz.py file in ceviche to manually enter the min and max values:

```python
def real(val, DK_max=None, DK_min=None, DK_title=None, outline=None, ax=None, cbar=False, cmap='RdBu', outline_alpha=0.5):
    """Plots the real part of 'val', optionally overlaying an outline of 'outline'
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    if DK_max is None:
        vmax = np.abs(val).max()
    else:
        vmax = DK_max
    
    if DK_min is None:
        vmin = -vmax
    else:
        vmin = DK_min

    h = ax.imshow(np.real(val.T), cmap=cmap,
                  origin='lower', vmin=vmin, vmax=vmax)

    if outline is not None:
        ax.contour(outline.T, 0, colors='k', alpha=outline_alpha)

    ax.set_ylabel('y')
    ax.set_xlabel('x')
    if DK_title is not None:
        ax.set_title(DK_title)
    if cbar:
        plt.colorbar(h, ax=ax, orientation="horizontal")

    return ax


def abs(val, DK_max=None, DK_title=None, outline=None, ax=None, cbar=False, cmap='magma', outline_alpha=0.5, outline_val=None):
    """Plots the absolute value of 'val', optionally overlaying an outline of 'outline'
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    if DK_max is None:
        vmax = np.abs(val).max()
    else:
        vmax = DK_max

    h = ax.imshow(np.abs(val.T), cmap=cmap, origin='lower', vmin=0, vmax=vmax)

    if outline_val is None and outline is not None:
        outline_val = 0.5*(outline.min()+outline.max())
    if outline is not None:
        ax.contour(outline.T, [outline_val], colors='w', alpha=outline_alpha)

    ax.set_ylabel('y')
    ax.set_xlabel('x')
    if DK_title is not None:
        ax.set_title(DK_title)
    if cbar:
        plt.colorbar(h, ax=ax,  orientation="horizontal")

    return ax
```