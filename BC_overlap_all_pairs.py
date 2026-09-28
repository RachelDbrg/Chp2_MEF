import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os


def plot_overlap_from_csv(
    csv_path,
    output_dir,
    variables=("C", "M", "P", "D1"),
    output_name="pairwise_overlaps"
):
    """
    Compute pairwise overlap indices from a CSV and plot them over time.

    Args:
        csv_path (str): Path to input CSV file
        variables (tuple/list): Variables to compare
        output_dir (str): Directory to save the figure
        output_name (str): Name of the output image file
    """


    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(csv_path)

    
    # -----------------------------
    # Prepare output directory
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Generate variable pairs
    # -----------------------------
    #pairs = list(itertools.combinations(variables, 2))

    # -----------------------------
    # Keep only species present initially
    # -----------------------------
    time_values = sorted(data["TimeStep"].unique())
    initial_time = time_values[0]

    data_init = data[data["TimeStep"] == initial_time]

    active_variables = [
        var for var in variables
        if var in data.columns and np.nanmax(data_init[var]) > 0
    ]

    print("Active species:", active_variables)

    # Generate pairs ONLY from active species
    pairs = list(itertools.combinations(active_variables, 2))

    TimeStep_values = []
    overlap_dict = {pair: [] for pair in pairs}


    # -----------------------------
    # Compute overlaps
    # -----------------------------
    for t, group in data.groupby("TimeStep"):
        TimeStep_values.append(t)

        for var1, var2 in pairs:
            if var1 not in group.columns or var2 not in group.columns:
                overlap_dict[(var1, var2)].append(np.nan)
                continue

            X = group[var1].values
            Y = group[var2].values

            # Remove negative noise
            X = np.maximum(X, 0)
            Y = np.maximum(Y, 0)

            num = np.sum(np.sqrt(X * Y))
            denom = np.sqrt(np.sum(X)) * np.sqrt(np.sum(Y))

            overlap = (num / denom) * 100 if denom > 0 else np.nan
            overlap_dict[(var1, var2)].append(overlap)


    # -----------------------------
    # Sort by timestep
    # -----------------------------
    TimeStep_values = np.array(TimeStep_values)
    order = np.argsort(TimeStep_values)

    TimeStep_values = TimeStep_values[order]

    for k in overlap_dict:
        overlap_dict[k] = np.array(overlap_dict[k])[order]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for (var1, var2), values in overlap_dict.items():
        label = f"{var1}-{var2}"
        plt.plot(TimeStep_values, values, marker='o', label=label)

    plt.xlabel("TimeStep")
    plt.ylabel("Overlap Index (%)")
    plt.title("Pairwise Overlap Indices Through Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    

    output_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot → {output_path}")