import astropy
from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize
import numpy as np  
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_fits(
    file_path: str,
    print_header: bool = False, 
):
    """ Read in the data from a FITS file."""
    hdul = fits.open(file_path)
    file = hdul[0].data
    header = hdul[0].header
    hdul.close()
    
    if print_header:
        print(repr(header))

    return file


def plot_fits(
    data: np.ndarray,
    title: str,
    norm: LogNorm = LogNorm(),
    flip_xaxis: bool = False,
):
    """ Plot the data from a FITS file."""
    if data is not None:
        # Handle byte-swapping if needed
        # data = np.nan_to_num(data, nan=0.0)
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(20, 6))

        im = ax.imshow(
            data, 
            cmap='gray', 
            origin='lower', 
            norm=norm, 
            aspect='auto'
        )
        ax.set_title(title)
        ax.set_xlabel('X Pixels')
        ax.set_ylabel('Y Pixels')
        if flip_xaxis:
            ax.invert_xaxis()

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad = 0.025)
        cbar.set_label('Pixel Intensity')

    else:
        print("No data found in the FITS file.")

def create_inset(arc_spectrum, ax, start, end, x, y):
    """ Create an inset plot in the main plot."""
    ax_inset = inset_axes(
        ax, 
        width="15%", 
        height="30%", 
        bbox_to_anchor=(x, y, 1, 1), 
        bbox_transform=ax.transAxes, 
        loc='upper left'
    )
    ax_inset.plot(
        arc_spectrum / np.max(arc_spectrum[start:end]), 
        color='black'
    )
    ax_inset.set_xlim(start, end)
    ax_inset.set_yticks([])
    ax_inset.set_ylim(0, 1)
    # ax_inset.tick_params(axis='x', direction='out', pad=-15)
    # ax_inset.tick_params(left=False, right=False, labelleft=False)
        

def spectrum(
    data: np.ndarray,
    position: int, 
    width: int, 
    negative_vals: bool = False,    
):
    """ Extract a spectrum from the data given a Y position and width."""
    # Take a vertical slice of the data
    slice_data = data[position:position + width, :]
    # Sum the pixel values along the vertical axis
    spectrum = np.sum(slice_data, axis=0)
    # Remove negative values
    if negative_vals:
        spectrum[spectrum < 0] = 0
    return spectrum


def create_master_bias(
    bias_files: list,
):
    """ Create a master bias frame from a list of bias files."""
    # Read in the bias files
    bias_data = []
    for file in bias_files:
        with fits.open(file) as hdul:
            bias_data.append(hdul[0].data)
    
    # Combine the bias frames
    master_bias = np.median(bias_data, axis=0)
    return master_bias


def create_master_dark(
    dark_files: list,
    master_bias: np.ndarray,
    #exposure_times: list,
):
    """ Create a master dark frame from a list of dark files."""
    # Read in the dark files
    dark_data = []
    for file in dark_files:
        with fits.open(file) as hdul:
            dark_data.append(hdul[0].data)
    
    # Subtract the master bias from the dark frames
    dark_data = [np.array(dark) - master_bias for dark in dark_data]

    # Combine the dark frames
    master_dark = np.median(np.array(dark_data), axis=0)
    return master_dark


def create_master_flat(
    flat_files: list,
    master_bias: np.ndarray,
):
    """ Create a master flat frame from a list of flat files."""
    # Read in the flat files
    flat_data = []
    for file in flat_files:
        with fits.open(file) as hdul:
            flat_data.append(hdul[0].data)
    
    # Subtract the master bias from the flat frames
    flat_data = [np.array(flat) - master_bias for flat in flat_data]
    
    # Combine the flat frames
    master_flat = np.median(np.array(flat_data), axis=0)
    return master_flat

def find_maximum_peak_in_range(
    data: np.ndarray,
    start: int,
    end: int,
):
    """ Find the maximum peak and corresponding pixel in the data."""
    # Find the maximum value in the data
    max_value = np.max(data[start:end])
    # Find the position of the maximum value
    peak = np.argmax(data[start:end])
    return max_value, start + peak

