import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter


def plot_gradient_heatmaps(
    csv_path,
    output_dir,
    species_list=("C", "M", "P"),
    output_name="gradient_magnitudes_heatmap.png",
    chunksize=500_000
):
    """
    Plot heatmaps of gradient magnitudes for each species and timestep,
    compatible with long-format CSV (x, y, TimeStep, Field, Value).
    """

    # ------------------------------------------------------------
    # STEP 1 — Identify timesteps WITHOUT loading full CSV
    # ------------------------------------------------------------
    time_values_all = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["TimeStep"]):
        time_values_all.extend(chunk["TimeStep"].unique())

    time_values_all = sorted(set(time_values_all))

    if not time_values_all:
        print("No timesteps found.")
        return

    time_values = [
        time_values_all[0],
        time_values_all[len(time_values_all)//2],
        time_values_all[-1]
    ]

    labels = ["Initial", "Middle", "Final"][:len(time_values)]

    print("Selected timesteps:", time_values)

    # ------------------------------------------------------------
    # STEP 2 — Prepare output directory
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # STEP 3 — Figure setup
    # ------------------------------------------------------------
    n_rows = len(time_values)
    n_cols = len(species_list)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5*n_cols, 4*n_rows),
        squeeze=False
    )

    # ------------------------------------------------------------
    # STEP 4 — Loop over timesteps
    # ------------------------------------------------------------
    for i, t in enumerate(time_values):

        print(f"\nProcessing timestep {t}")

        # Collect only this timestep
        df_list = []
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            df_list.append(chunk[chunk["TimeStep"] == t])

        df_t_long = pd.concat(df_list, ignore_index=True)
        df_t_long.columns = df_t_long.columns.str.strip()

        # --------------------------------------------------------
        # Convert long → wide for this timestep
        # --------------------------------------------------------
        df_t = df_t_long.pivot_table(
            index=["y", "x"],
            columns="Field",
            values="Value",
            aggfunc="first"
        ).reset_index()

        x_unique = np.sort(df_t["x"].unique())
        y_unique = np.sort(df_t["y"].unique())

        # --------------------------------------------------------
        # Compute gradient magnitudes
        # --------------------------------------------------------
        for sp in species_list:
            g0 = f"grad_tot_{sp}_0"
            g1 = f"grad_tot_{sp}_1"
            g2 = f"grad_tot_{sp}_2"

            if all(col in df_t.columns for col in [g0, g1, g2]):
                df_t[f"grad_mag_{sp}"] = np.sqrt(
                    df_t[g0].fillna(0)**2 +
                    df_t[g1].fillna(0)**2 +
                    df_t[g2].fillna(0)**2
                )
            else:
                df_t[f"grad_mag_{sp}"] = np.zeros(len(df_t))

        # --------------------------------------------------------
        # Plot heatmaps
        # --------------------------------------------------------
        for j, sp in enumerate(species_list):

            ax = axes[i, j]

            Z = (
                df_t
                .pivot_table(
                    index="y",
                    columns="x",
                    values=f"grad_mag_{sp}",
                    aggfunc="first"
                )
                .reindex(index=y_unique, columns=x_unique)
                .values
            )

            Z_smooth = gaussian_filter(Z, sigma=1)

            im = ax.imshow(
                Z_smooth,
                origin="lower",
                extent=[x_unique.min(), x_unique.max(),
                        y_unique.min(), y_unique.max()],
                cmap="inferno",
                aspect="equal"
            )

            ax.contour(
                x_unique,
                y_unique,
                Z_smooth,
                levels=8,
                colors="white",
                linewidths=0.5
            )

            title = f"{sp}"
            if j == 0:
                title = f"{labels[i]} (t={t})\n{sp}"

            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved heatmap figure → {output_path}")
