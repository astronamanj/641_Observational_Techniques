import astropy
import numpy as np  

from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize

from scipy import stats, fft
from scipy.signal import gaussian
from scipy.ndimage import uniform_filter1d

from rich.pretty import Pretty
from sigpyproc.readers import FilReader

import os, sys, builtins, contextlib
from contextlib import contextmanager

freqs = np.linspace(800, 400, 1024)

@contextmanager
def suppress_sigpyproc():
    orig_open  = builtins.open
    orig_write = os.write

    def fake_open(path, *args, **kwargs):
        if path == '/dev/tty':
            return orig_open(os.devnull, 'w')
        return orig_open(path, *args, **kwargs)

    def fake_write(fd, data):
        if fd in (1, 2):
            return len(data)   # swallow output
        return orig_write(fd, data)

    builtins.open = fake_open
    os.write      = fake_write

    try:
        yield
    finally:
        builtins.open = orig_open
        os.write      = orig_write

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
    pulsar_array, 
    title: str = 'Pulsar Source Waterfall Plot', 
):
    fig = plt.figure(figsize=(10, 6))
    im = plt.imshow(pulsar_array, aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label('ADC units', fontsize=14)
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


def plot_timeseries(
    pulsar_32, 
    pulsar_32_array, 
    freq_mask,
    type : str = "mean", 
    title: str = 'Pulsar Time Series'
):
    fig = plt.figure(figsize=(10, 6))
    if type == "mean":
        plt.plot(
            np.arange(pulsar_32.header.nsamples) * pulsar_32.header.tsamp,
            np.nanmean(pulsar_32_array[~freq_mask, :], axis=0)
        )
    elif type == "sum":
        plt.plot(
            np.arange(pulsar_32.header.nsamples) * pulsar_32.header.tsamp,
            np.nansum(pulsar_32_array[~freq_mask, :], axis=0)
        )
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
    plt.ylabel('Boxcar Width (samples)', fontsize=16)
    plt.tight_layout()
    plt.show()

def get_correlations(results): 
    correlations = [np.max(data["corr"]) for data in results.values()]
    print(f"Ratio of max with median: {np.max(correlations)/np.median(correlations)}")
    print(f"Ratio of max with mean: {np.max(correlations)/np.mean(correlations)}")

    upper_bound = np.percentile(correlations, 99)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=100, alpha=0.7, color='blue', edgecolor='black')
    # plt.axvline(lower_bound, color='red', linestyle='--', label='99% Confidence Interval')
    plt.axvline(upper_bound, color='red', linestyle='--', label='99% Confidence Interval')
    plt.axvline(np.max(correlations), color='red', linestyle='-', linewidth=2, label='Max Correlation')
    plt.xlabel('Correlation Values', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    # plt.title('Histogram of Correlations', fontsize=16)
    plt.legend(fontsize=14)
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
        tfactor = 32, 
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

def get_noise_stats(
    pulsar_data_32, 
    freq_mask
):
    """
    Compute the noise statistics of the pulsar data.  
    """
    # Compute the baseline (median along frequency axis for each time bin)
    baseline = np.nansum(pulsar_data_32.data[~freq_mask, :], axis=0)

    # Subtract the baseline from the original data to get the zeroed data
    zeroed = pulsar_data_32.data - np.nanmean(pulsar_data_32.data[~freq_mask, :], axis=0)

    # Calculate the noise standard deviation along the frequency axis for each time bin
    noise_std = np.std(zeroed, axis=0)

    # Scale the data by the noise standard deviation, guarding against division by zero
    scaled_data = zeroed / np.where(noise_std == 0, 1, noise_std)

    # Plot 1: Original data, baseline-subtracted data, noise std, and scaled data
    plt.figure(figsize=(12, 8))

    # Original data
    plt.subplot(2, 2, 1)
    plt.imshow(pulsar_data_32.data, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Original Data")

    # Baseline-subtracted (zeroed) data
    plt.subplot(2, 2, 2)
    plt.imshow(zeroed, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Baseline-Subtracted Data")

    # Noise standard deviation as a function of time
    plt.subplot(2, 2, 3)
    plt.plot(noise_std, marker='o', linestyle='-')
    plt.xlabel("Time Bin")
    plt.ylabel("Noise Std")
    plt.title("Noise Standard Deviation vs Time")

    # Scaled data (noise normalized)
    plt.subplot(2, 2, 4)
    plt.imshow(scaled_data, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Scaled Data (Noise Normalized)")

    plt.tight_layout()
    plt.show()

    # Histogram of scaled data values to inspect distribution
    # plt.figure(figsize=(6, 4))
    # plt.hist(scaled_data.flatten(), bins=50, density=True)
    # plt.xlabel("Scaled Data Value")
    # plt.ylabel("Density")
    # plt.title("Histogram of Scaled Data")
    # plt.show()

    return baseline, zeroed, noise_std


def get_best_parameters(
    results : dict,
    metric : str, 
    tsamp : float, 
):
    (best_dm, best_width), best_data = max(
        results.items(),
        key=lambda kv: np.max(np.array(kv[1][metric], dtype=np.float32))
    )
    best_snr = best_data["sn"]
    best_time = best_data["midtimes"]

    print(f"Best DM: {best_dm}, Best Width: {best_width * tsamp}, Max SNR: {best_snr}")
    return best_dm, best_width, best_snr


def boxcart_fit(
    data_32, 
    data_data_32, 
    freq_mask,
    dm_range, 
    width_range, 
    baseline, 
    noise_std
):
    results = {}
    
    for dm in dm_range:
        # Dedisperse the injected data and subtract the baseline
        baseline = np.average(np.nansum(data_data_32.data[~freq_mask, :], axis=0))
        time_series = data_32.dedisperse(dm).data
        zeroed = time_series - baseline  

        for width in width_range:
            # Compute the metrics
            averages = uniform_filter1d(zeroed, size=width, mode='nearest')
            squared_averages = uniform_filter1d(zeroed**2, size=width, mode='nearest')
            stds = np.sqrt(np.abs(squared_averages - averages**2))
            sums = np.convolve(zeroed, np.ones(width), mode='valid')
            total_noise = noise_std * width
            snr = sums / (total_noise * np.sqrt(width))

            # Compute midtimes for the windowed sums
            midtimes = (np.arange(sums.size) + (width - 1) / 2) * data_32.header.tsamp

            # Store computed metrics in the results dictionary
            results[(dm, width)] = {
                "midtimes": midtimes,
                "sums": sums,
                "averages": averages[:len(sums)],  
                "stds": stds[:len(sums)],  
                "sn": snr, 
            }
    return results


def gaussian_fit(
    data_32, 
    data_data_32, 
    freq_mask,
    dm_range, 
    width_range, 
):
    results = {}

    for dm in dm_range:
        # Dedisperse the injected data and subtract the baseline
        baseline = np.average(np.nansum(data_data_32.data[~freq_mask, :], axis=0))
        time_series = data_32.dedisperse(dm).data
        zeroed = time_series - baseline  
        noise_std = np.std(zeroed, axis=0)

        for width in width_range:
            # Create Gaussian template
            gauss_template = gaussian(len(zeroed), std=width)
            gauss_template *= np.max(zeroed) 

            # Compute correlation with Gaussian template
            corr = np.correlate(zeroed, gauss_template, mode='full')

            # Compute the snr
            weighted_integral = np.dot(np.abs(zeroed), gauss_template)
            norm_factor = noise_std * np.sqrt(np.sum(gauss_template**2))
            snr = weighted_integral / norm_factor

            # Compute midtimes for the windowed sums
            midtimes = (np.arange(corr.size) + (len(zeroed) - 1) / 2) * data_32.header.tsamp

            # Store computed metrics in the results dictionary
            results[(dm, width)] = {
                "midtimes": midtimes,
                "corr": corr,
                "sn": snr, 

            }

    return results