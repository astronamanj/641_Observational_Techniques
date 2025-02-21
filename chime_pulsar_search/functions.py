import astropy
import numpy as np  
from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize

from scipy import stats, fft

from rich.pretty import Pretty
from sigpyproc.readers import FilReader

# matplot_kwargs = {
#         'fontsize': 18,
#         'figure.figsize': (10, 6),
#         'axes.labelsize': 14,
#         'axes.titlesize': 16,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 12,
#         'legend.loc': 'best',
#         'grid.color': 'gray',
#         'grid.linestyle': '--',
#         'grid.linewidth': 0.5
# }

def flux_density(
        freq: float,
        ref_freq: float = 400,
        ref_flux: float = 8.684,
        spectral_index: float = -0.6,
)-> float:
    """Calculate the flux density in Jy"""
    return ref_flux * (freq / ref_freq) ** spectral_index

def normalize_data_per_channel(
        data_array: np.ndarray, 
        mask: np.ndarray, 
        invert_mask: bool = False, 
):
    if invert_mask:
        mask = ~mask

    # Compute mean and standard deviation per channel (across time bins), filtering out bad channels
    mean = np.mean(data_array[mask, :], axis=1)
    std = np.std(data_array[mask, :], axis=1)

    # Normalize only the valid frequency channels
    norm_array = np.copy(data_array)
    norm_array[mask, :] = (data_array[mask, :] - mean[:, np.newaxis]) / std[:, np.newaxis]
    norm_array[mask, :] = np.nan

    return norm_array