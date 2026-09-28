import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def compute_mass_and_plot(csv_path, output_dir, chunksize=1_000_000):
    """
    Memory-safe mass computation for huge PDE CSV files.
    Processes the CSV in chunks so it never gets killed.
    """

    os.makedirs(output_dir, exist_ok=True)

    # First pass: detect dx, dy from small chunk
    first_chunk = next(pd.read_csv(csv_path, chunksize=chunksize))
    x_unique = np.sort(first_chunk['x'].unique())
    y_unique = np.sort(first_chunk['y'].unique())

    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]

    print(f"Detected dx = {dx}, dy = {dy}")

    species = ["C", "M", "P", "D1"]

    # Dictionary of dictionaries: species → timestep → mass
    mass_accumulator = {sp: defaultdict(float) for sp in species}

    # Stream the CSV in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk[chunk["Field"].isin(species)]

        # Multiply values by cell area
        chunk["mass"] = chunk["Value"] * dx * dy

        # Accumulate mass per timestep per species
        for sp in species:
            sub = chunk[chunk["Field"] == sp]
            grouped = sub.groupby("TimeStep")["mass"].sum()

            for t, m in grouped.items():
                mass_accumulator[sp][t] += m

    # Convert accumulators to pandas Series
    mass_dict = {
        sp: pd.Series(mass_accumulator[sp]).sort_index()
        for sp in species
    }

    # Plot each species
    for sp, series in mass_dict.items():
        plt.figure(figsize=(8, 5))
        plt.plot(series.index, series.values, linewidth=2)
        plt.title(f"Total Mass Through Time — Species {sp}")
        plt.xlabel("TimeStep")
        plt.ylabel("Total Mass")
        plt.grid(True)

        png_path = os.path.join(output_dir, f"mass_{sp}.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()

    return mass_dict
