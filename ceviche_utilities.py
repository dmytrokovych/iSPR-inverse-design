import numpy as np
import autograd.numpy as npa
from autograd.scipy.signal import convolve as conv
from skimage.draw import circle
import matplotlib.pyplot as plt
import os
import imageio

# constants

EPSILON_0 = 8.85418782e-12              # vacuum permittivity
MU_0 = 1.25663706e-6                    # vacuum permeability
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)     # speed of light in vacuum
Q_e = 1.602176634e-19                   # funamental charge


# utility functions

def mask_combine_rho(rho, bg_rho, design_region):
    """Utility function for combining the design region rho and the background rho
    """
    return rho*design_region + bg_rho*(design_region == 0).astype(float)


def operator_proj(rho, eta=0.5, beta=100, N=1):
    """Density projection
    eta     : Center of the projection between 0 and 1
    beta    : Strength of the projection
    N       : Number of times to apply the projection
    """
    for i in range(N):
        rho = npa.divide(npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)),
                         npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

    return rho


def _create_blur_kernel(radius):
    """Helper function used below for creating the conv kernel"""
    rr, cc = circle(radius, radius, radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=float)
    kernel[rr, cc] = 1
    return kernel/kernel.sum()


def operator_blur(rho, radius=2, N=1):
    """Blur operator implemented via two-dimensional convolution
    radius    : Radius of the circle for the conv kernel filter
    N         : Number of times to apply the filter

    Note that depending on the radius, the kernel is not always a
    perfect circle due to "pixelation" / stair casing
    """

    kernel = _create_blur_kernel(radius)

    for i in range(N):
        # For whatever reason HIPS autograd doesn't support 'same' mode, so we need to manually crop the output
        rho = conv(rho, kernel, mode='full')[radius:-radius, radius:-radius]

    return rho


def epsr_parameterization(rho, bg_rho, design_region, epsr_min, epsr_max, radius=2, N_blur=1, beta=100, eta=0.5, N_proj=1):
    """Defines the parameterization steps for constructing rho
    """

    # Combine rho and bg_rho
    rho = mask_combine_rho(rho, bg_rho, design_region)

    rho = operator_blur(rho, radius=radius, N=N_blur)
    rho = operator_proj(rho, beta=beta, eta=eta, N=N_proj)

    # Final masking undoes the blurring
    rho = mask_combine_rho(rho, bg_rho, design_region)

    return epsr_min + (epsr_max-epsr_min) * rho


def animate(val, max=None, title=None, outline=None, cbar=False, cmap='RdBu', outline_alpha=0.5, gif_name='mygif', frames_num=24):
    filenames = []
    for frame in range(frames_num):
        # plot the frmae
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        if max is None:
            vmax = np.abs(val).max()
        else:
            vmax = max

        h = ax.imshow(np.real(val.T * np.exp(1j*2*np.pi*frame/frames_num)), cmap=cmap,
                  origin='lower', vmin=-vmax, vmax=vmax)

        if outline is not None:
            ax.contour(outline.T, 0, colors='k', alpha=outline_alpha)

        ax.set_ylabel('y')
        ax.set_xlabel('x')

        if title is not None:
            ax.set_title(title)
        if cbar:
            plt.colorbar(h, ax=ax, orientation="horizontal")
        
        # create file name and append it to a list
        filename = f'./gif/frames/{frame}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()

    # build gif
    with imageio.get_writer(f'./gif/{gif_name}.gif', mode='I') as writer:
        print('Creating gif...\n')
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        print('Gif saved\n')
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename) 