"""
===============================================================================
DATA LOADER MODULE
===============================================================================
This module generates synthetic datasets for testing search and indexing
algorithms (B-Tree, Linear Regression Index, etc.).

The DatasetGenerator class provides functions to create datasets ranging from
1 to 1,000,000 keys in different distributions:
    • Sequential — ordered numbers (best-case data)
    • Uniform — random spread across a range
    • Mixed — clustered/random blend to simulate real-world skew

These datasets are used as input for performance benchmarking, testing how
well various index structures handle different data patterns.

Usage:
    from src.utils.data_loader import DatasetGenerator

    keys = DatasetGenerator.generate_uniform(10000)
    print(keys[:10])  # Preview data

Next Steps:
    - Add CSV save/load utilities if you want to persist datasets.
    - Extend with skewed or Zipf distributions for more complex tests.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    """Generate small test datasets for Phase 1 B-Tree benchmarks."""

    @staticmethod
    def generate_uniform(size: int, min_val: int = 0, max_val: int = 1_000_000) -> np.ndarray:
        """Uniformly distributed random keys."""
        keys = np.random.uniform(min_val, max_val, size)
        return np.sort(keys)

    @staticmethod
    def generate_sequential(size: int, start: int = 0, step: int = 1) -> np.ndarray:
        """Sequential keys (0, 1, 2, …)."""
        return np.arange(start, start + size * step, step, dtype=np.float64)

    @staticmethod
    def generate_mixed(size: int) -> np.ndarray:
        """Mixed distribution: uniform + two clusters."""
        uniform = np.random.uniform(0, 1_000_000, int(size * 0.4))
        cluster1 = np.random.normal(250_000, 10_000, int(size * 0.3))
        cluster2 = np.random.normal(750_000, 10_000, int(size * 0.3))
        keys = np.concatenate([uniform, cluster1, cluster2])
        return np.sort(np.unique(keys))[:size]


# -----------------------------------------------------------------------------
# Quick-run tester: save + show interactive plots
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating datasets and plotting\n")

    size = 100_000  # 100k points per plot

    seq = DatasetGenerator.generate_sequential(size)
    uniform = DatasetGenerator.generate_uniform(size)
    mixed = DatasetGenerator.generate_mixed(size)

    print("Datasets generated.")

    # show each plot
    def plot_data(data, title):
        plt.figure(figsize=(10, 4))
        plt.plot(data, '.', markersize=1)
        plt.title(f"{title} (100,000 points)")
        plt.xlabel("Index")
        plt.ylabel("Key Value")
        plt.tight_layout()
        plt.show()  

    # Plot and show each dataset
    plot_data(seq, "Sequential Dataset")
    plot_data(uniform, "Uniform Dataset")
    plot_data(mixed, "Mixed Dataset")

    print("\nAll plots saved and displayed.")
