"""
This module contains the functions used in the CHIME pulsar search project. 

Functions:
    flux_density: Calculate the flux density in Jy
    normalize_data_per_channel: Normalize the data per channel
    plot_waterfall: Plot the waterfall plot of the data
    plot_mean_spectrum: Plot the mean spectrum of the data
    get_time_series: Get the time series of the data
    plot_timeseries: Plot the time series of the data
    plot_heatmap: Plot the heatmap of the data
    read_and_downsize_data: Read and downsize the data
    gaussian_fit: Fit a Gaussian to the data
"""

import astropy
import numpy as np 
import random 

from astropy.io import fits

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize

from scipy import stats, fft
from scipy.signal import gaussian
from scipy.ndimage import uniform_filter1d

from rich.pretty import Pretty
from sigpyproc.readers import FilReader

freqs = np.linspace(800, 400, 1024)
dm_range = np.linspace(-200, -5, 200) 
width_range = np.arange(4, 200, 4)    
dm_bin_width = dm_range[1] - dm_range[0]
tfactor = 8


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

    mean = np.mean(data_array[mask, :], axis=1)
    std = np.std(data_array[mask, :], axis=1)

    norm_array = np.copy(data_array)
    norm_array[mask, :] = (data_array[mask, :] - mean[:, np.newaxis]) / std[:, np.newaxis]
#     norm_array[mask, :] = data_array[mask, :] / mean[:, np.newaxis]

    return norm_array

def plot_waterfall(
    data_array, 
    freq_mask = None,
    title: str = 'Pulsar Source Waterfall Plot', 
    vmin : float = None, 
    vmax : float = None
):
    fig = plt.figure(figsize=(10, 6))
    if freq_mask is None:
        freq_mask = np.zeros(data_array.shape[0], dtype=bool
    )
    im = plt.imshow(data_array[~freq_mask, :], aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label('ADC units', fontsize=14)
    if vmin is not None and vmax is not None:
        im.set_clim(vmin, vmax)
    else:
        # cbar.set_clim(np.nanmin(data_array), np.nanmax(data_array))
        pass
    plt.title(title, fontsize=14)
    plt.xlabel('Time [samples]', fontsize=18)
    plt.ylabel('Freq. [channel]', fontsize=18)
    plt.show()

def plot_mean_spectrum(
    pulsar, 
    pulsar_masked, 
    freq_mask,
    title: str = 'Mean Spectrum',
):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        pulsar_masked.header.chan_freqs, 
        pulsar.chan_stats.mean, 
    )
    plt.plot(
        pulsar_masked.header.chan_freqs, 
        np.where(~freq_mask, pulsar_masked.chan_stats.mean, np.nan), 
        linewidth=2, 
    )
    plt.ylabel('Avg. ADC [ul]', fontsize=18)
    plt.xlabel('Freq [MHz]', fontsize=18)
    plt.title('RFI Masking', fontsize=14)
    plt.legend(['Original', 'Masked'])
    plt.show()


def get_time_series(
    source_data_array, 
    freq_mask,
    type : str = "mean"
):
    if freq_mask is None:
        freq_mask = np.zeros(source_data_array.shape[0], dtype=bool)
    if type == "mean":
        return np.nanmean(source_data_array[~freq_mask, :], axis=0)
    elif type == "sum":
        return np.nansum(source_data_array[~freq_mask, :], axis=0)


def plot_timeseries(
    source_data_array, 
    freq_mask,
    type : str = "mean", 
    title: str = 'Pulsar Time Series'
):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(get_time_series(source_data_array, freq_mask, type))
    plt.ylabel(f'{type} ADC [ul]', fontsize=18)
    plt.xlabel('Time [s]', fontsize=18)
    plt.title(title, fontsize=14)
    plt.show()


def plot_heatmap(
    results : dict, 
    metric : str = "corr", 
):
    dm_values    = np.sort(list({dm for dm, width in results.keys()}))
    width_values = np.sort(list({width for dm, width in results.keys()}))

    heatmap = np.zeros((len(width_values), len(dm_values)))

    for i, width in enumerate(width_values):
        for j, dm in enumerate(dm_values):
            heatmap[i, j] = np.max(
                np.array(results[(dm, width)][metric], dtype=np.float32)
            )

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        heatmap,
        aspect='auto',
        interpolation='nearest',
        extent=[dm_values.min(), dm_values.max(),
                width_values.min(), width_values.max()],
        origin='lower'
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation Strength", fontsize=14)
    plt.xlabel('Dispersion Measure (DM)', fontsize=16)
    plt.ylabel('Width (samples)', fontsize=16)
    plt.tight_layout()
    plt.show()


def read_and_downsize_data(
    file : str = "./data/data_261146047.fil",
    method : str = "mad",
    threshold : int = 3,
    tfactor : int = 32,
    object_type : str = "Pulsar",
    do_plots : bool = True,
    **kwargs
):
    # Read in the data
    pulsar = FilReader(file) 
    Pretty(pulsar.header)

    pulsar.compute_stats()
    pulsar_data = pulsar.read_block(0, pulsar.header.nsamples, pulsar.header.fch1, pulsar.header.nchans)
    pulsar_array = pulsar_data.data

    # Plot the waterfall
    if do_plots:
        plot_waterfall(pulsar_array, title=f'{object_type} Source Waterfall Plot')

    # Mask the RFI
    outfile_name = file.replace(".fil", "_masked.fil")
    _, chan_mask = pulsar.clean_rfi(
        method=method,
        threshold=threshold,
        outfile_name=outfile_name,
    )
    pulsar_masked = FilReader(outfile_name) 
    pulsar_masked.compute_stats()

    freq_mask = pulsar_masked.chan_stats.mean == 0

    # Plot the mean spectrum
    if do_plots:
        plot_mean_spectrum(pulsar, pulsar_masked, freq_mask, title=f'{object_type} Mean Spectrum')

    # Downsample the data
    outfile_name = outfile_name.replace(".fil", "_f1_t32.fil")
    pulsar_masked.downsample(
        tfactor = tfactor, 
        outfile_name = outfile_name
    )
    pulsar_32 = FilReader(outfile_name) 
    pulsar_32.compute_stats()
    pulsar_data_32 = pulsar_32.read_block(0, pulsar_32.header.nsamples, pulsar_32.header.fch1, pulsar_32.header.nchans)
    pulsar_32_array = pulsar_data_32.data

    # Waterfall of downsampled data
    if do_plots:
        plot_waterfall(pulsar_32_array, title=f'{object_type} Data Downsampled Waterfall Plot')

    return pulsar_32, pulsar_data_32, freq_mask


def gaussian_fit(
    source_data, 
    freq_mask,
    dm_range, 
    width_range, 
):
    results = {}

    for dm in dm_range:
        # Dedisperse the injected data and subtract the baseline
        dedispersed = source_data.dedisperse(dm).data
        time_series = get_time_series(dedispersed, freq_mask, "sum")
        # time_series -= np.median(time_series)

        for width in width_range:
            # Create Gaussian template
            gauss_template = gaussian(len(time_series), std=width)
            # gauss_template /= np.sum(gauss_template)  # Normalize template to sum=1

            # Compute correlation with Gaussian template
            corr = np.convolve(time_series, gauss_template, mode='same')

            # Compute midtimes for the windowed sums
            midtimes = np.arange(len(time_series)) * source_data.header.tsamp

            # Store computed metrics in the results dictionary
            results[(dm, width)] = {
                "midtimes": midtimes,
                "corr": np.abs(corr),
            }

    return results