import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spatial_snapshots(
    csv_path,
    output_dir,
    species_list=("C", "M", "P", "D1"),
    output_name="combined_plot"
):
    """
    Create spatial plots and diagonal profiles for selected timesteps.

    Args:
        csv_path (str): Path to CSV file
        species_list (list/tuple): Species to consider
        output_dir (str): Directory to save the figure
        output_name (str): Output image filename
    """

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(csv_path)

    # -----------------------------
    # Output directory
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    needed_columns = ["x", "y", "TimeStep"] + list(species_list)

    # -----------------------------
    # Select timesteps (initial, middle, final)
    # -----------------------------
    time_values_all = sorted(df["TimeStep"].unique())

    if len(time_values_all) == 0:
        print("No TimeStep data found.")
        return

    time_values = [
        time_values_all[0],
        time_values_all[len(time_values_all) // 2],
        time_values_all[-1]
    ]

    time_values = sorted(set(time_values))  # remove duplicates if few steps

    # -----------------------------
    # Filter active species
    # -----------------------------
    # Identify the initial timestep
    initial_time = time_values_all[0]

    # Filter dataframe at initial timestep
    df_init = df[df["TimeStep"] == initial_time]

    # Keep only species that are present (density > 0) initially
    active_species = [
        sp for sp in species_list
        if sp in df.columns and np.nanmax(df_init[sp]) > 0
    ]

    if not active_species:
        print("No active species found.")
        return

    print("Active species:", active_species)

    # -----------------------------
    # Create subplot grid
    # -----------------------------
    n_times = len(time_values)
    n_sp = len(active_species)
    n_cols = n_sp + 1  # last column = diagonal plot

    fig, axes = plt.subplots(
        n_times, n_cols,
        figsize=(4 * n_cols, 3 * n_times),
        squeeze=False
    )

    # -----------------------------
    # Loop over timesteps
    # -----------------------------
    for i, time_value in enumerate(time_values):

        df_t = df[df["TimeStep"] == time_value]

        x_unique = np.sort(df_t["x"].unique())
        y_unique = np.sort(df_t["y"].unique())

        def create_grid(species):
            Z = df_t.pivot_table(
            index="y",
            columns="x",
            values=species,
            aggfunc="first"   # or "first", "sum", etc.
            )

            
            return Z.values

        # ---- Spatial plots ----
        for j, sp in enumerate(active_species):
            ax = axes[i, j]
            Z = create_grid(sp)

            im = ax.imshow(
                Z,
                origin="lower",
                extent=[x_unique.min(), x_unique.max(),
                        y_unique.min(), y_unique.max()],
                aspect='auto'
            )


            # ---- Contours (20%, 50%, 70%, 90%) ----
            Z_max = np.nanmax(Z)

            if Z_max > 0:
                levels = [0.2 * Z_max, 0.5 * Z_max, 0.7 * Z_max, 0.9 * Z_max]

                X, Y = np.meshgrid(x_unique, y_unique)

                cs = ax.contour(
                    X, Y, Z,
                    levels=levels,
                    colors='white',
                    linewidths=1
                )

                ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")


            # Diagonal line
            ax.plot(
                [x_unique.min(), x_unique.max()],
                [y_unique.min(), y_unique.max()],
                'w--', linewidth=1
            )

            if i == 0:
                ax.set_title(sp)

            if j == 0:
                label = ["Initial", "Middle", "Final"][i] if i < 3 else f"t={time_value}"
                ax.set_ylabel(f"{label} (t={time_value})\n y")

            ax.set_xlabel("x")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- Diagonal plot ----
        ax_diag = axes[i, -1]

        tol = (x_unique.max() - x_unique.min()) / len(x_unique)
        diag_df = df_t[np.abs(df_t["x"] - df_t["y"]) < tol]
        diag_df = diag_df.sort_values("x")

        for sp in active_species:
            ax_diag.plot(diag_df["x"], diag_df[sp], label=sp)

        if i == 0:
            ax_diag.set_title("Diagonal")
            ax_diag.legend(fontsize=8)

        if i == len(time_values) - 1:
            ax_diag.set_xlabel("x")

        ax_diag.set_ylabel("Concentration")
        ax_diag.grid()

    # -----------------------------
    # Save figure
    # -----------------------------
    plt.tight_layout()

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure → {output_path}")



    