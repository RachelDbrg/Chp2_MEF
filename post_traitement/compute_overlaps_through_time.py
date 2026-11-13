import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import itertools
#from main_figures_bis import load_wide_csv

def compute_overlap_snapshots(
    wide_csv_path,
    variables=("C", "M", "P", "D"),
    n_snapshots=5
):

    data = pd.read_csv(wide_csv_path)

    time_values = np.sort(data["TimeStep"].unique())

    # indices des snapshots
    idx = np.linspace(
        0,
        len(time_values) - 1,
        n_snapshots,
        dtype=int
    )

    snapshot_times = time_values[idx]

    initial_time = time_values[0]

    data_init = data[data["TimeStep"] == initial_time]

    active_variables = [
        var for var in variables
        if var in data.columns
        and np.nanmax(data_init[var]) > 0
    ]

    pairs = list(itertools.combinations(active_variables, 2))

    overlaps = []

    for t in snapshot_times:

        group = data[data["TimeStep"] == t]

        pair_overlaps = []

        for var1, var2 in pairs:

            X = np.maximum(group[var1].values, 0)
            Y = np.maximum(group[var2].values, 0)

            num = np.sum(np.sqrt(X * Y))
            denom = np.sqrt(np.sum(X)) * np.sqrt(np.sum(Y))

            if denom > 0:
                pair_overlaps.append(
                    100 * num / denom
                )

        overlaps.append(np.mean(pair_overlaps))

    return snapshot_times, overlaps