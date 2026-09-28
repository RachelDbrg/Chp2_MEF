from matplotlib import animation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def create_species_gifs(
    csv_path,
    output_dir,
    species_list=("C")
):

    # ------------------------------------------------------------
    # Load ONLY the useful columns
    # ------------------------------------------------------------
    needed_columns = ["x", "y", "TimeStep"] + list(species_list)
    df = pd.read_csv(csv_path, usecols=needed_columns)

    # Strip column names (important if CSV has trailing spaces)
    df.columns = df.columns.str.strip()

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Time values
    # ------------------------------------------------------------
    time_values = sorted(df["TimeStep"].unique())

    # ------------------------------------------------------------
    # Filter valid species
    # ------------------------------------------------------------
    species_candidates = [sp for sp in species_list if sp in df.columns]

    # Keep only species present initially
    t0 = time_values[0]
    df_init = df[df["TimeStep"] == t0]

    active_species = [
        sp for sp in species_candidates
        if not np.allclose(df_init[sp], 0, equal_nan=True)
    ]

    print("Active species:", active_species)

    if len(active_species) < 1:
        print("Not enough species.")
        return

    # ------------------------------------------------------------
    # Create one GIF per species
    # ------------------------------------------------------------
    for species in active_species:

        vmax = df[species].max()

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()

            t = time_values[frame]
            df_t = df[df["TimeStep"] == t]

            x_unique = np.sort(df_t["x"].unique())
            y_unique = np.sort(df_t["y"].unique())

            # Pivot-table grid (handles triangular domains)
            Z = df_t.pivot_table(
                index="y",
                columns="x",
                values=species,
                aggfunc="first"
            ).values

            im = ax.imshow(
                Z,
                origin="lower",
                extent=[x_unique.min(), x_unique.max(),
                        y_unique.min(), y_unique.max()],
                vmin=0,
                vmax=vmax,
                aspect="auto"
            )

            ax.set_title(f"{species} at t={t}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            return [im]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(time_values),
            blit=False
        )

        gif_path = os.path.join(output_dir, f"{species}.gif")
        ani.save(gif_path, writer="pillow", fps=5)

        plt.close()
        print(f"Saved GIF → {gif_path}")
