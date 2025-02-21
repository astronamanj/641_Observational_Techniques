import astropy
import numpy as np  
from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize

from scipy import stats

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