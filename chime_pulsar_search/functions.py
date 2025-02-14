import astropy
import numpy as np  
from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize

from rich.pretty import Pretty
from sigpyproc.readers import FilReader

def flux_density(
        freq: float,
        ref_freq: float = 400,
        ref_flux: float = 8.684,
        spectral_index: float = -0.6,
)-> float:
    """Calculate the flux density in Jy"""
    return ref_flux * (freq / ref_freq) ** spectral_index