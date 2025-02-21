# Generating synthetic data for model testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SEED=42
np.random.seed(SEED) # Set the random seed for reproducibility

# ===== Data Generation Functions =====

def gen_exp_decay_pattern(
        x0: float,
        decay_rate: float,
        sigma: float,
        interval: tuple = (0, 80),
        resolution: int = 1000,
) -> np.array:
    """
    Generates the base pattern for synthetic XRD data using an exponentially decaying function with Gaussian noise.

    Parameters:
        x0 (float): Initial value of the base pattern.
        decay_rate (float): Rate of exponential decay.
        sigma (float): Standard deviation of Gaussian noise added at each step.
        interval (tuple): The interval of the base pattern (x-axis range).
        resolution (int): Number of steps to generate the process.

    Returns:
        x_values (np.array): The x-values for the base pattern.
        base_pattern (np.array): The base pattern with exponential decay and noise.
    """
    dt = (interval[1] - interval[0]) / resolution  # Step size
    x_values = np.linspace(interval[0], interval[1], resolution)
    
    # Exponential decay function
    base_pattern = x0 * np.exp(-decay_rate * x_values)

    # Add Gaussian noise at each step
    noise = np.random.normal(0, sigma, size=resolution)
    base_pattern += noise
    
    return x_values, base_pattern

def gen_base_pattern(
        x0: float,
        kappa: float,
        theta: float,
        sigma: float,
        interval: tuple = (0, 80),
        resolution: int = 1000,
) -> np.array:
    """
    Generates the base pattern for synthetic XRD data via the Ornstein-Uhlenbeck process.

    Parameters:
        interval: tuple, the interval of the base pattern (x-axis range)
        x0: float, the initial value of the base pattern
        kappa: float, the mean reversion coefficient (controls how quickly values revert to theta)
        theta: float, the long-term mean of the process
        sigma: float, the volatility (controls noise amplitude)
        num_steps: int, the number of steps to generate the process
    
    Returns:
        base_pattern: np.array, the base pattern for synthetic XRD data
    """
    dt = (interval[1] - interval[0]) / resolution  # "Time" step
    # x values over the specified interval
    x_values = np.linspace(interval[0], interval[1], resolution)
    
    # Initialize the base pattern array
    base_pattern = np.zeros(resolution)
    base_pattern[0] = x0  # Set the initial value
    
    # Generate the Ornstein-Uhlenbeck process
    for i in range(1, resolution):
        dW = np.random.normal(0, np.sqrt(dt))  # Wiener process increment
        base_pattern[i] = base_pattern[i-1] + kappa * (theta - base_pattern[i-1]) * dt + sigma * dW
    
    return x_values, base_pattern

def gen_reference_patterns(num_patterns: int = 10, interval: tuple = (0, 80)) -> list:
    """
    Generates reference peak patterns without noise.

    Parameters:
        num_patterns: int, the number of reference patterns to generate (default: 10).
        interval: tuple, the interval of the base pattern (x-axis range)

    Returns:
        reference_patterns: list of tuples, each containing (peak_positions, peak_heights, peak_widths).
    """
    reference_patterns = []
    
    for _ in range(num_patterns):
        num_peaks = np.random.choice([1, 2, 3, 4, 5])  # Uniform selection from 1 to 5
        peak_positions = np.sort(np.random.uniform(interval[0], interval[1], num_peaks))  # Positions
        peak_heights = np.abs(np.random.normal(loc=1.0, scale=0.4, size=num_peaks))  # Gaussian heights, positive values
        peak_widths = np.random.uniform(0.01, 0.5, num_peaks)  # Uniformly sampled widths
        
        reference_patterns.append((peak_positions, peak_heights, peak_widths))
    
    return reference_patterns

def gen_experimental_pattern(
        base_pattern: np.array,
        reference_patterns: list,
        interval: tuple = (0, 80),
        resolution: int = 1000,
        phase_shift_prob: float = 0.3,  # Probability of applying a phase shift
        phase_shift_range: float = 10.0,  # Maximum shift in peak positions
        height_factor: float = 5.0 # Scaling peak height
) -> tuple:
    """
    Inserts random peaks from two reference patterns into the base pattern.

    Parameters:
        base_pattern (np.array): The base pattern generated via the Ornstein-Uhlenbeck process.
        reference_patterns (list): List of tuples (peak_positions, peak_heights, peak_widths).
        interval (tuple): The x-axis range of the pattern.
        resolution (int): Number of points in the generated XRD pattern.
        phase_shift_prob (float): Probability that a peak will be phase-shifted.
        phase_shift_range (float): Maximum phase shift range (in x-axis units).

    Returns:
        x_values (np.array): The x-values of the XRD pattern.
        xrd_pattern (np.array): The base pattern with peaks inserted.
    """
    x_values = np.linspace(interval[0], interval[1], resolution)
    xrd_pattern = base_pattern.copy()

    # Select two random reference patterns to mix
    idx1, idx2 = np.random.choice([i for i in range(len(reference_patterns))], 2, replace=False)
    ref1, ref2 = reference_patterns[idx1], reference_patterns[idx2]

    # Extract peaks
    combined_peaks = list(zip(*ref1)) + list(zip(*ref2))  # combined the unpacked tuples

    for peak_pos, peak_height, peak_width in combined_peaks:
        # Random phase shift
        if np.random.rand() < phase_shift_prob:
            shift = np.random.uniform(-phase_shift_range, phase_shift_range)
            peak_pos += shift  

        # Ensure peak remains within the valid range
        if interval[0] <= peak_pos <= interval[1]:
            # Add Gaussian peak to the pattern
            xrd_pattern += height_factor * peak_height * np.exp(-0.5 * ((x_values - peak_pos) / peak_width) ** 2)


    return x_values, xrd_pattern

def gen_total_pattern(
        interval: tuple =(0, 2*np.pi),
        x0: float =1000,
)->np.array:
    """
    Generates the total pattern for synthetic XRD data.

        1. Generates the base pattern via the Ornstein-Uhlenbeck process.
        2. Inserts random peaks into the base pattern.

    parameters:
        interval: tuple, the interval of the base pattern. (default: (0, 2*np.pi))
        x0: float, the initial value of the base pattern. (default: 1000)
        num_peaks: int, the number of peaks to insert into the base pattern
    
    returns:
        total_pattern: np.array, the total pattern for synthetic XRD data
    """
    pass

# ===== Plotting Functions =====

def plot_single_reference_pattern(
        peak_positions: np.array,
        peak_heights: np.array,
        peak_widths: np.array,
        interval: tuple = (0, 80),
        resolution = 1000
):
    """
    Plotting a single reference pattern. XRD patterns, with the correct orientation and no noise.
    """

    x = np.linspace(interval[0],interval[1],resolution)
    y = np.zeros_like(x)

    # Add peaks to base
    for i in range(len(peak_positions)):
        y += peak_heights[i] * np.exp(-0.5 * ((x - peak_positions[i]) / peak_widths[i])**2) # Gaussian peaks

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("2θ")
    plt.ylabel("Intensity")
    plt.title("Reference Pattern for Synthetic XRD Data")
    plt.show()

def plot_all_reference_patterns(
    reference_patterns, 
    num_cols=2,
    interval=(0, 80),
    resolution=1000,
):
    """
    Plots multiple reference XRD patterns as subplots in a single Matplotlib window.

    Parameters:
        reference_patterns (list): A list of tuples containing (peak_positions, peak_heights, peak_widths).
        num_cols (int): Number of columns in the subplot grid.
    """
    num_patterns = len(reference_patterns)
    num_rows = (num_patterns + num_cols - 1) // num_cols  # Compute the required rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten axes array for easy indexing

    for i, (peak_positions, peak_heights, peak_widths) in enumerate(reference_patterns):
        x = np.linspace(interval[0], interval[1], resolution)  # High-resolution x-axis
        y = np.zeros_like(x)

        # Generate Gaussian peaks
        for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
            y += height * np.exp(-0.5 * ((x - pos) / width) ** 2)

        # Plot in the corresponding subplot
        axes[i].plot(x, y)
        axes[i].set_title(f"Pattern {i+1}")
        axes[i].set_xlabel("2θ", fontsize=6)
        axes[i].set_ylabel("Intensity", fontsize=6)

    # Hide unused subplots if num_patterns is not a perfect multiple of num_cols
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_experimental_pattern(
        x_values: np.array,
        base_pattern: np.array,
        xrd_pattern: np.array,
):
    """
    Plots the experimental XRD pattern with the base pattern and inserted peaks.

    Parameters:
        x_values (np.array): The x-values of the XRD pattern.
        base_pattern (np.array): The base pattern generated via the Ornstein-Uhlenbeck process.
        xrd_pattern (np.array): The base pattern with peaks inserted.
        interval (tuple): The x-axis range of the pattern.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, base_pattern, label="Base Pattern", alpha=0.5)
    plt.plot(x_values, xrd_pattern, label="Experimental Pattern", color="red", alpha=0.5)
    plt.xlabel("2θ")
    plt.ylabel("Intensity")
    plt.title("Synthetic XRD Data")
    plt.legend()
    plt.show()


# ===== Dataset Generating Function =====

def gen_dataset(
        num_references: int = 10,
        num_experimentals: int = 1000,
        interval: tuple = (0, 80),
        resolution: int = 1000,
        noise_levels: list = [0.1, 0.2, 0.3, 0.5],
        peak_scale_factors: list = [2.0, 3.0, 5.0]
)->dict:
    """
    Generates a synthetic XRD dataset with separate metadata and pattern storage.

    Parameters:
        num_references (int): Number of reference patterns to generate.
        num_experimentals (int): Number of experimental patterns to generate.
        interval (tuple): The x-axis range for XRD patterns.
        resolution (int): Number of data points per pattern.
        noise_levels (list): Different levels of Gaussian noise to apply.
        peak_scale_factors (list): Scaling factors for peak heights.

    Saves:
        - "synthetic_xrd_metadata.csv" (Metadata file)
        - "synthetic_xrd_patterns.npy" (NumPy array storing all XRD patterns)
    """

    # Generate reference patterns
    reference_patterns = gen_reference_patterns(num_patterns=num_references, interval=interval)
    # Save reference patterns
    np.save("data/reference_patterns.npy", reference_patterns)

    # Store experimental patterns & metadata
    experimental_patterns = np.zeros((num_experimentals, resolution))
    metadata = []

    for i in range(num_experimentals):
        # Randomized parameters
        sigma = np.random.choice(noise_levels)
        height_factor = np.random.choice(peak_scale_factors)
        decay_rate = np.random.uniform(0.02, 0.08)
        x0 = np.random.uniform(3, 10)

        # Generate base pattern
        x_values, base_pattern = gen_exp_decay_pattern(
            x0=x0,
            decay_rate=decay_rate,
            sigma=sigma,
            interval=interval,
            resolution=resolution
        )

        # Select reference patterns
        ref_id1, ref_id2 = np.random.choice(range(num_references), 2, replace=False)

        # Generate experimental pattern
        phase_shift_prob = np.random.uniform(0.2, 0.5)
        phase_shift_range = np.random.uniform(5, 15)

        x_values, xrd_pattern = gen_experimental_pattern(
            base_pattern,
            [reference_patterns[ref_id1], reference_patterns[ref_id2]],
            interval=interval,
            resolution=resolution,
            phase_shift_prob=phase_shift_prob,
            phase_shift_range=phase_shift_range,
            height_factor=height_factor
        )

        # Store pattern
        experimental_patterns[i, :] = xrd_pattern

        # Store metadata
        metadata.append({
            "pattern_id": i,
            "ref_id1": ref_id1,
            "ref_id2": ref_id2,
            "noise_level": sigma,
            "peak_scale": height_factor,
            "decay_rate": decay_rate,
            "x0": x0,
            "phase_shift_prob": phase_shift_prob,
            "phase_shift_range": phase_shift_range
        })

    # Save metadata as CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv("data/synthetic_xrd_metadata.csv", index=False)

    # Save patterns as a NumPy array
    np.save("data/synthetic_xrd_patterns.npy", experimental_patterns)

    print("Dataset generated and saved!")

    return {"metadata": metadata, "experimental_patterns": experimental_patterns}


if __name__=="__main__":

    # For testing/generation of synthetic data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="plot_base_pattern",choices=["plot_base_pattern", "plot_single_reference_pattern", "plot_all_reference_patterns", "plot_experimental_pattern", "plot_exp_decay_pattern", "gen_dataset"], help="The mode to run the script in")

    args = parser.parse_args()

    match args.mode:
        case "plot_base_pattern":
            import matplotlib.pyplot as plt

            # Generate the base pattern
            # TODO : Settle on the parameters for generating base patterns
            interval = (0, 80)
            x0 = 20
            kappa = 0.05 # Mean reversion coefficient
            theta = 14 # Long-term mean
            sigma = 0.7 # Volatility
            x_values, base_pattern = gen_base_pattern(x0, kappa, theta, sigma)

            # Plot the base pattern
            plt.plot(x_values, base_pattern)
            plt.ylim(0, 2*max(base_pattern))
            plt.xlim(0, interval[1])
            plt.xlabel("2θ")
            plt.ylabel("Intensity")
            plt.title("Base Pattern for Synthetic XRD Data")
            plt.show()

        case "plot_single_reference_pattern":
            # Generate reference patterns
            reference_patterns = gen_reference_patterns(num_patterns=10)
            
            peak_positions, peak_heights, peak_widths = reference_patterns[0]

            print(f"Peak positions: {peak_positions}")
            print(f"Peak heights: {peak_heights}")
            print(f"Peak widths: {peak_widths}")

            # Plot the first reference pattern
            plot_single_reference_pattern(peak_positions, peak_heights, peak_widths)

        case "plot_all_reference_patterns":
            # Generate reference patterns
            reference_patterns = gen_reference_patterns(num_patterns=10)
            plot_all_reference_patterns(reference_patterns)

        case "plot_experimental_pattern":

            # Generate the base pattern
            interval = (0, 80)
            x0 = 20
            kappa = 0.05
            theta = 14
            sigma = 0.7
            # x_values, base_pattern = gen_base_pattern(x0, kappa, theta, sigma)
            x_values, base_pattern = gen_exp_decay_pattern(x0=5, decay_rate=0.05, sigma=0.2)
            # Get reference patterns
            reference_patterns = gen_reference_patterns(num_patterns=10)
            # Generate the experimental pattern
            x_values, xrd_pattern = gen_experimental_pattern(base_pattern, reference_patterns)
            # Plot the experimental pattern
            plot_experimental_pattern(x_values, base_pattern, xrd_pattern)

        case "plot_exp_decay_pattern":
            x_values, base_pattern = gen_exp_decay_pattern(x0=5, decay_rate=0.05, sigma=0.2)
            plt.plot(x_values, base_pattern)
            plt.ylim(0, 2*max(base_pattern))
            plt.xlim(0, 80)
            plt.xlabel("2θ")
            plt.ylabel("Intensity")
            plt.title("Base Pattern for Synthetic XRD Data")
            plt.show()

        case "gen_dataset":
            num_experimentals = 1000
            dataset = gen_dataset(num_references=10, num_experimentals=num_experimentals)
            print(d)
            
            df = pd.DataFrame([
                {
                    "ref_id1": ref_ids[0],
                    "ref_id2": ref_ids[1],
                    "noise_level": noise_level,
                    "peak_scale": peak_scale,
                    "xrd_pattern": xrd_pattern
                }
                for _, xrd_pattern, ref_ids, noise_level, peak_scale in dataset["experimental_patterns"]
            ])

            # Save dataset
            df.to_csv("data/synthetic_xrd_dataset.csv", index=False)

        case _:
            raise ValueError(f"Invalid mode: {args.mode}")

    pass