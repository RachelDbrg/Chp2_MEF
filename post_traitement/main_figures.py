from convert_vtk_csv import vtu_to_csv
from heterogeneous_landscape import plot_maps_from_csv
from BC_overlap_all_pairs import plot_overlap_from_csv
from plot_overlaps_heatmap import plot_spatial_overlaps_grid
from memory_safe_2D_concentration import plot_spatial_snapshots
from plot_surface_through_time import plot_surface_through_time
from overlap_through_time import temporal_density_change
from create_gif_through_time import create_species_gifs
from equilibrium_time import func_equilibrium_time
from test_3D_resources_plot import plot_gradient_heatmaps
from check_mass_conservation import compute_mass_and_plot
from compute_overlaps_through_time import compute_overlap_snapshots

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

# =============================================================================
# SETTINGS
# =============================================================================

simulation_folder = (
    "/users/rdubourg/Documents/Chp2_MEF/"
    "simulations/var_parms/i/Null_NMP_/"
)

output_csv = (
    "/users/rdubourg/Documents/Chp2_MEF/figures/"
    "overlap_strategy_snapshots.csv"
)


# =============================================================================
# PARAMETER READING
# =============================================================================

def read_parameters(parameter_file):

    parms = {}

    with open(parameter_file, "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("==="):
                continue

            if " = " not in line:
                continue

            key, value = line.split(" = ", 1)

            try:
                parms[key.strip()] = float(
                    value.strip()
                )
            except ValueError:
                pass

    return parms


# =============================================================================
# STRATEGY METRICS
# =============================================================================

def extract_strategy_metrics(scenario_dir):

    parameter_file = os.path.join(
        scenario_dir,
        "parameters.txt"
    )

    parms = read_parameters(parameter_file)

    return {

        "c_ifd":
            parms["c1"] + parms["c2"],

        "c_lof":
            parms["c3"]
            + parms["c4"]
            + parms["c5"]
            + parms["c6"],

        "m_ifd":
            parms["m1"] + parms["m2"],

        "m_lof":
            parms["m3"]
            + parms["m4"]
            + parms["m5"]
            + parms["m6"],

        "p_ifd":
            parms["p1"] + parms["p2"],

        "p_lof":
            parms["p3"]
            + parms["p4"]
            + parms["p5"]
            + parms["p6"],

        "d_ifd":
            parms["d1"] + parms["d2"],

        "d_lof":
            parms["d3"]
            + parms["d4"]
            + parms["d5"]
            + parms["d6"]
    }


# =============================================================================
# OVERLAP - Batt index
# =============================================================================

def overlap_metric(X, Y):

    X = np.maximum(X, 0)
    Y = np.maximum(Y, 0)

    denom = (
        np.sqrt(np.sum(X))
        * np.sqrt(np.sum(Y))
    )

    if denom <= 0:
        return np.nan

    return (
        100
        * np.sum(np.sqrt(X * Y))
        / denom
    )


# =============================================================================
# FIND SCENARIOS
# =============================================================================

scenario_dirs = sorted(
    glob.glob(
        os.path.join(
            simulation_folder,
            "scenario_*"
        )
    )
)

print(
    f"{len(scenario_dirs)} scenarios found"
)

results = []


# =============================================================================
# CREATE WIDE CSV WITH SELECTED SNAPSHOTS
# =============================================================================

def create_wide_snapshot_csv(
    csv_path,
    output_path,
    n_snapshots=5
):

    df_long = pd.read_csv(csv_path)

    all_times = np.sort(
        df_long["TimeStep"].unique()
    )

    n_snapshots = min(
        n_snapshots,
        len(all_times)
    )

    idx = np.linspace(
        0,
        len(all_times) - 1,
        n_snapshots
    ).round().astype(int)

    idx = np.unique(idx)

    selected_times = all_times[idx]

    print(
        "Selected times:",
        selected_times
    )

    df_subset = df_long[
        df_long["TimeStep"].isin(
            selected_times
        )
    ]

    df_wide = (
        df_subset
        .pivot_table(
            index=["x", "y", "TimeStep"],
            columns="Field",
            values="Value"
        )
        .reset_index()
    )

    df_wide.columns.name = None

    df_wide.to_csv(
        output_path,
        index=False
    )

    return selected_times




# =============================================================================
# LOOP OVER SCENARIOS
# =============================================================================

for scenario_dir in scenario_dirs:

    scenario_name = os.path.basename(
        scenario_dir
    )

    print(
        f"\nProcessing {scenario_name}"
    )

    # -------------------------------------------------------------------------
    # Read strategy metrics
    # -------------------------------------------------------------------------

    metrics = extract_strategy_metrics(
        scenario_dir
    )

    # -------------------------------------------------------------------------
    # Convert VTU -> long CSV
    # -------------------------------------------------------------------------

    vtu_to_csv(
        scenario_dir,
        "PDE_output",
        n_snapshots=5
)

    csv_path = os.path.join(
        scenario_dir,
        "PDE_output.csv"
    )

    if not os.path.exists(csv_path):

        print(
            f"Missing {csv_path}"
        )

        continue

    # -------------------------------------------------------------------------
    # Build reduced wide CSV with only 5 snapshots
    # -------------------------------------------------------------------------

    wide_csv = os.path.join(
        scenario_dir,
        "PDE_output_wide_5snapshots.csv"
    )

    selected_times = create_wide_snapshot_csv(
        csv_path,
        wide_csv,
        n_snapshots=5
    )

    df = pd.read_csv(
        wide_csv
    )

    # -------------------------------------------------------------------------
    # Compute overlaps
    # -------------------------------------------------------------------------

    for snapshot_id, t in enumerate(
        selected_times
    ):

        group = df[
            df["TimeStep"] == t
        ]

        if len(group) == 0:
            continue

        overlap_cm = overlap_metric(
            group["C"].values,
            group["M"].values
        )

        overlap_cp = overlap_metric(
            group["C"].values,
            group["P"].values
        )

        overlap_mp = overlap_metric(
            group["M"].values,
            group["P"].values
        )

        results.append(
        {
            "scenario": scenario_name,
            "snapshot": snapshot_id,
            "time": t,
            "overlap_CM": overlap_cm,
            "overlap_CP": overlap_cp,
            "overlap_MP": overlap_mp,
            **metrics
        }
            )


df_results = pd.DataFrame(results)

print(df_results.head())

df_results.to_csv(
    output_csv,
    index=False
)

print(f"Saved: {output_csv}")



