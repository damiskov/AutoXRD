import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d

# ====== Collection of smoothing functions ======

def moving_average_smooth(pattern, window_size=5):
    return uniform_filter1d(pattern, size=window_size, mode='nearest')

def moving_average_smooth_v2(pattern, window_size=5):
    """
    Applies a moving average filter to smooth an XRD pattern while handling edge effects.

    Parameters:
        pattern (np.array): The XRD intensity values.
        window_size (int): Number of neighboring points to average.

    Returns:
        smoothed_pattern (np.array): Smoothed XRD pattern.
    """
    pad_width = window_size // 2
    padded_pattern = np.pad(pattern, pad_width, mode='edge')  # Extend with edge values
    smoothed_pattern = np.convolve(padded_pattern, np.ones(window_size) / window_size, mode='valid')
    
    return smoothed_pattern

def savgol_smooth(pattern, window_size=9, poly_order=3, mode='interp'):
    """
    Applies Savitzky-Golay filter to smooth an XRD pattern.

    Parameters:
        pattern (np.array): The XRD intensity values.
        window_size (int): Number of neighboring points (must be odd).
        poly_order (int): Polynomial order for fitting.

    Returns:
        smoothed_pattern (np.array): Smoothed XRD pattern.
    """
    return savgol_filter(pattern, window_size, poly_order, mode=mode) # TODO: Change mode='interp' to mode='same'?

def gaussian_smooth(pattern, sigma):
    """
    Applies Gaussian smoothing to an XRD pattern.

    Parameters:
        pattern (np.array): The XRD intensity values.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        smoothed_pattern (np.array): Smoothed XRD pattern.
    """
    return gaussian_filter1d(pattern, sigma=sigma)