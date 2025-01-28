import astropy
from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm
import numpy as np  
from scipy.signal import find_peaks

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
):
    """ Plot the data from a FITS file."""
    if data is not None:
        # Handle byte-swapping if needed
        # data = np.nan_to_num(data, nan=0.0)
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(15, 5))

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

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad = 0.025)
        cbar.set_label('Pixel Intensity')

        plt.show()
    else:
        print("No data found in the FITS file.")

        

def spectrum(
    data: np.ndarray,
    position: int, 
    width: int, 
):
    """ Extract a spectrum from the data given a Y position and width."""
    # Take a vertical slice of the data
    slice_data = data[position:position + width, :]
    # Sum the pixel values along the vertical axis
    spectrum = np.sum(slice_data, axis=0)
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
    max_value = np.max(data)
    # Find the position of the maximum value
    peak = np.argmax(data[start:end])
    return max_value, peak

def find_peaks_in_threshold(
    data: np.ndarray,
    threshold_low: float,
    threshold_high: float,
):
    """ Find peaks in the data above a certain threshold."""
    # Find the peaks above the threshold
    # Identify indices where the data is within the threshold range
    valid_indices = np.where((data > threshold_low) & (data < threshold_high))[0]
    
    # Create a mask for the valid range to zero out data outside the threshold
    masked_data = np.zeros_like(data)
    masked_data[valid_indices] = data[valid_indices]
    
    # Find peaks in the masked data
    peaks, _ = find_peaks(masked_data)
    
    # Get positions of the peaks
    positions = peaks
    
    return peaks, positions

def pixel_to_wavelength(
    lambda0: float = 3053.5651855469,  
    dlambda: float = 0.25,  
    ref_pix: int = 1 
):
    """ Convert pixel number to wavelength."""
    return lambda0 + (ref_pix - 1) * dlambda

def wavelength_to_pixel(
    lambda0: float = 3053.5651855469,  
    dlambda: float = 0.25,  
    wavelength: float = 0.0 
):
    """ Convert wavelength to pixel number."""
    return int((wavelength - lambda0) / dlambda) + 1
