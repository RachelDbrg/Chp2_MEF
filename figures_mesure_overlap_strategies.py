import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compute_overlaps_through_time import (
    compute_overlap_snapshots
)

# =============================================================================
# SETTINGS
# =============================================================================

simulation_folder = (
    "/users/rdubourg/Documents/Chp2_MEF/"
    "simulations/var_parms/i/Null_NMP_/"
)

figures_folder = (
    "/users/rdubourg/Documents/Chp2_MEF/figures"
)

csv_name = "PDE_output"


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


def extract_strategy_metrics(
    scenario_dir
):

    parms = read_parameters(
        os.path.join(
            scenario_dir,
            "parameters.txt"
        )
    )

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
            + parms["d6"],
    }


# =============================================================================
# LOOP OVER SCENARIOS
# =============================================================================

results = []

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

for scenario_dir in scenario_dirs:

    scenario_name = os.path.basename(
        scenario_dir
    )

    print(
        f"Processing {scenario_name}"
    )

    csv_path = os.path.join(
        scenario_dir,
        f"{csv_name}.csv"
    )

    if not os.path.exists(csv_path):
        continue

    metrics = extract_strategy_metrics(
        scenario_dir
    )

    try:

        times, overlaps = (
            compute_overlap_snapshots(
                csv_path,
                n_snapshots=5
            )
        )

    except Exception as e:

        print(
            f"Overlap failed for "
            f"{scenario_name}: {e}"
        )

        continue

    if len(overlaps) == 0:
        continue

    for i, (t, ov) in enumerate(
        zip(times, overlaps)
    ):

        results.append(
            {
                "scenario":
                    scenario_name,

                "snapshot":
                    i,

                "time":
                    t,

                "overlap":
                    ov,

                **metrics
            }
        )


# =============================================================================
# DATAFRAME
# =============================================================================

df = pd.DataFrame(results)

print(df.head())

print(
    f"{len(df)} snapshots"
)


# =============================================================================
# SAVE CSV
# =============================================================================

output_csv = os.path.join(
    figures_folder,
    "overlap_strategy_snapshots.csv"
)

df.to_csv(
    output_csv,
    index=False
)

print(
    f"Saved: {output_csv}"
)

