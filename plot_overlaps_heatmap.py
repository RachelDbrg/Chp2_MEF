def plot_spatial_overlaps_grid(
    csv_path,
    output_dir="overlap_plots",
    species_list=("C", "M", "P", "D1"),
    output_name="overlap_grid"
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    import os

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Filter valid species
    # -----------------------------
    species_candidates = [sp for sp in species_list if sp in df.columns]

    # Keep only species present initially
    t0 = df["TimeStep"].min()
    df_init = df[df["TimeStep"] == t0]

    active_species = [
        sp for sp in species_candidates
        if not np.allclose(df_init[sp], 0, equal_nan=True)
    ]

    print("Active species:", active_species)

    if len(active_species) < 2:
        print("Not enough species.")
        return

    # -----------------------------
    # Generate pairs
    # -----------------------------
    pairs = list(itertools.combinations(active_species, 2))
    print("Pairs:", pairs)

    # -----------------------------
    # Select timesteps
    # -----------------------------
    time_values_all = sorted(df["TimeStep"].unique())

    time_values = [
        time_values_all[0],
        time_values_all[len(time_values_all)//2],
        time_values_all[-1]
    ]
    time_values = sorted(set(time_values))

    # -----------------------------
    # Create subplot grid
    # -----------------------------
    n_rows = len(time_values)
    n_cols = len(pairs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False
    )

    # -----------------------------
    # Loop over time and pairs
    # -----------------------------
    for i, timestep in enumerate(time_values):
        df_t = df[df["TimeStep"] == timestep]

        x_unique = np.sort(df_t["x"].unique())
        y_unique = np.sort(df_t["y"].unique())

        def grid(sp):
            Z = df_t.pivot_table(
                index="y",
                columns="x",
                values=sp,
                aggfunc="first"
            )
            return np.nan_to_num(Z.values, nan=0.0)

        label = ["Initial", "Middle", "Final"][i] if i < 3 else f"t={timestep}"

        for j, (var1, var2) in enumerate(pairs):

            ax = axes[i, j]

            X = grid(var1)
            Y = grid(var2)

            overlap = np.sqrt(X * Y)

            im = ax.imshow(
                overlap,
                origin="lower",
                extent=[x_unique.min(), x_unique.max(),
                        y_unique.min(), y_unique.max()],
                aspect="auto"
            )

            # Column titles
            if i == 0:
                ax.set_title(f"{var1}-{var2}")

            # Row labels
            if j == 0:
                ax.set_ylabel(f"{label}\n y")

            ax.set_xlabel("x")

            # Optional: contour lines
            Zmax = np.nanmax(overlap)
            if Zmax > 0:
                levels = [0.2*Zmax, 0.5*Zmax, 0.7*Zmax, 0.9*Zmax]
                Xg, Yg = np.meshgrid(x_unique, y_unique)

                ax.contour(
                    Xg, Yg, overlap,
                    levels=levels,
                    colors="white",
                    linewidths=0.8
                )

    # -----------------------------
    # Shared colorbar
    # -----------------------------
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("√(X·Y) overlap")

    # -----------------------------
    # Save
    # -----------------------------
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved grid → {output_path}")