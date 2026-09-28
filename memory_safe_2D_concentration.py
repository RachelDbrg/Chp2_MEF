import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spatial_snapshots(
    csv_path,
    output_dir,
    species_list=("C", "M", "P", "D1"),
    output_name="combined_plot",
    chunksize=500_000
):
    """
    Create spatial plots and diagonal profiles for selected timesteps,
    adapted for long-format CSV (x, y, TimeStep, Field, Value),
    without ever loading the full CSV in memory.
    """

    # ------------------------------------------------------------
    # STEP 1 — Find all timesteps WITHOUT loading full CSV
    # ------------------------------------------------------------
    time_values_all = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["TimeStep"]):
        time_values_all.extend(chunk["TimeStep"].unique())

    time_values_all = sorted(set(time_values_all))

    if len(time_values_all) == 0:
        print("No TimeStep data found.")
        return

    # initial, middle, final
    time_values = [
        time_values_all[0],
        time_values_all[len(time_values_all) // 2],
        time_values_all[-1]
    ]
    time_values = sorted(set(time_values))

    print("Selected timesteps:", time_values)

    # ------------------------------------------------------------
    # STEP 2 — Determine active species (only initial timestep)
    # ------------------------------------------------------------
    df_init_list = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        df_init_list.append(chunk[chunk["TimeStep"] == time_values_all[0]])

    df_init_long = pd.concat(df_init_list, ignore_index=True)
    df_init_long.columns = df_init_long.columns.str.strip()

    active_species = []
    for sp in species_list:
        if sp in df_init_long["Field"].unique():
            if df_init_long[df_init_long["Field"] == sp]["Value"].max() > 0:
                active_species.append(sp)

    if not active_species:
        print("No active species found.")
        return

    print("Active species:", active_species)

    # ------------------------------------------------------------
    # STEP 3 — Prepare output directory
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # STEP 4 — Loop over timesteps
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        len(time_values),
        len(active_species) + 1,
        figsize=(4 * (len(active_species) + 1), 3 * len(time_values)),
        squeeze=False
    )

    for i, time_value in enumerate(time_values):

        print(f"\nProcessing timestep {time_value}...")

        # Collect only this timestep
        df_t_list = []
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            df_t_list.append(chunk[chunk["TimeStep"] == time_value])

        df_t_long = pd.concat(df_t_list, ignore_index=True)
        df_t_long.columns = df_t_long.columns.str.strip()

        # Convert long → wide for this timestep only
        df_t = df_t_long.pivot_table(
            index=["y", "x"],
            columns="Field",
            values="Value",
            aggfunc="first"
        ).reset_index()

        x_unique = np.sort(df_t["x"].unique())
        y_unique = np.sort(df_t["y"].unique())

        # --------------------------------------------------------
        # Spatial plots
        # --------------------------------------------------------
        for j, sp in enumerate(active_species):
            ax = axes[i, j]

            Z = df_t.pivot_table(
                index="y",
                columns="x",
                values=sp,
                aggfunc="first"
            ).values

            im = ax.imshow(
                Z,
                origin="lower",
                extent=[x_unique.min(), x_unique.max(),
                        y_unique.min(), y_unique.max()],
                aspect='auto'
            )

            Z_max = np.nanmax(Z)
            if Z_max > 0:
                levels = [0.2 * Z_max, 0.5 * Z_max, 0.7 * Z_max, 0.9 * Z_max]
                X, Y = np.meshgrid(x_unique, y_unique)
                #cs = ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=1)
                #ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

            ax.plot([x_unique.min(), x_unique.max()],
                    [y_unique.min(), y_unique.max()],
                    'w--', linewidth=1)


            species_labels = {
                "C": "Caribou",
                "P": "Wolf",
                "M": "Moose",
                "D1": "Deer"
                }


            if i == 0:
                ax.set_title(species_labels.get(sp, sp))


            if j == 0:
                label = ["Initial", "Middle", "Final"][i]
                ax.set_ylabel(f"{label} (t={time_value})\n y")

            ax.set_xlabel("x")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --------------------------------------------------------
        # Diagonal plot
        # --------------------------------------------------------
        ax_diag = axes[i, -1]

        tol = (x_unique.max() - x_unique.min()) / len(x_unique)
        diag_df = df_t[np.abs(df_t["x"] - df_t["y"]) < tol].sort_values("x")

        for sp in active_species:
            ax_diag.plot(diag_df["x"], diag_df[sp], label=sp)

        if i == 0:
            ax_diag.set_title("Diagonal")
            ax_diag.legend(fontsize=8)

        if i == len(time_values) - 1:
            ax_diag.set_xlabel("x")

        ax_diag.set_ylabel("Concentration")
        ax_diag.grid()

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved figure → {output_path}")
