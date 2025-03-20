import astropy
import numpy as np  
from scipy.ndimage import uniform_filter1d
from astropy.io import fits
from matplotlib import pyplot as plt  
from matplotlib.colors import LogNorm, Normalize

from scipy import stats, fft
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
    title: str = 'Pulsar Time Series'
):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(pulsar_32.header.nsamples) * pulsar_32.header.tsamp,
        np.nanmean(pulsar_32_array[~freq_mask, :], axis=0)
    )
    plt.ylabel('ADC [ul]', fontsize=18)
    plt.xlabel('Time [s]', fontsize=18)
    plt.title(title, fontsize=14)
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
    baseline = np.nanmean(pulsar_data_32.data[~freq_mask, :], axis=0)

    # Subtract the baseline from the original data to get the zeroed data
    zeroed = pulsar_data_32.data - baseline

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


def compute_score(
    data : dict, 
    alpha : float, 
    beta : float, 
    sn_min : float, 
    sn_max : float, 
    stds_min : float, 
    stds_max : float, 
):
    # Normalize SNR and noise std values
    sn_norm = (data["sn"] - sn_min) / (sn_max - sn_min) if sn_max > sn_min else 0
    stds_norm = (data["stds"] - stds_min) / (stds_max - stds_min) if stds_max > stds_min else 0
    return alpha * sn_norm + beta * stds_norm

def get_best_parameters(
    alpha : float, 
    beta : float, 
    results : dict,
    tsamp : float, 
    sn_min : float, 
    sn_max : float, 
    stds_min : float, 
    stds_max : float,
):
    # Find the best (dm, width) pair by maximizing the combined score
    (best_dm, best_width), best_data = max(
        results.items(), 
        key=lambda kv: np.max(compute_score(kv[1], alpha, beta, sn_min, sn_max, stds_min, stds_max))
    )
    
    # Locate the index of the maximum score within the best candidate
    scores = compute_score(best_data, alpha, beta, sn_min, sn_max, stds_min, stds_max)
    idx = np.argmax(scores)
    
    # Extract the corresponding best time and additional details
    best_time = best_data["midtimes"][idx]
    best_snr = best_data["sn"][idx]
    best_noise_std = best_data["stds"][idx]
    max_score = scores[idx]
    
    # Convert the best width to seconds using tsamp
    best_width_sec = best_width * tsamp
    
    return best_dm, best_width_sec, best_time, best_snr, best_noise_std, max_score


def compute_injection_results(
    inj_32, 
    dm_range, 
    width_range, 
    baseline, 
    noise_std
):
    results = {}
    
    for dm in dm_range:
        # Dedisperse the injected data and subtract the baseline
        time_series = inj_32.dedisperse(dm).data
        zeroed = time_series - baseline  

        for width in width_range:
            # Compute the metrics
            averages = uniform_filter1d(zeroed, size=width, mode='nearest')
            squared_averages = uniform_filter1d(zeroed**2, size=width, mode='nearest')
            stds = np.sqrt(np.abs(squared_averages - averages**2))
            sums = np.convolve(zeroed, np.ones(width), mode='valid')
            snr = sums / (noise_std * np.sqrt(width))

            # Compute midtimes for the windowed sums
            midtimes = (np.arange(sums.size) + (width - 1) / 2) * inj_32.header.tsamp

            # Store computed metrics in the results dictionary
            results[(dm, width)] = {
                "midtimes": midtimes,
                "sums": sums,
                "averages": averages[:len(sums)],  
                "stds": stds[:len(sums)],  
                "sn": snr
            }
    return results

def plot_heatmap(
    results : dict, 
    alpha : float,
    beta : float,
    sn_min : float,
    sn_max : float,
    stds_min : float,
    stds_max : float
):
    dm_values    = np.sort(list({dm for dm, width in results.keys()}))
    width_values = np.sort(list({width for dm, width in results.keys()}))

    heatmap = np.zeros((len(width_values), len(dm_values)))

    for i, width in enumerate(width_values):
        for j, dm in enumerate(dm_values):
            heatmap[i, j] = np.max(
                compute_score(
                    results[(dm, width)], 
                    alpha, 
                    beta, 
                    sn_min, 
                    sn_max, 
                    stds_min, 
                    stds_max
                )
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
    cbar.set_label("Combined Score", fontsize=14)
    plt.xlabel('Dispersion Measure (DM)', fontsize=16)
    plt.ylabel('Boxcar Width (samples)', fontsize=16)
    plt.tight_layout()
    plt.show()