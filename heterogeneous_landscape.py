import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_maps_from_csv(
    csv_path,
    output_dir,
    x_col="x",
    y_col="y",
    variables=("valeurs_paysage", "valeurs_diversity", "mec_tot_C", "grad_tot_C", "der_sec_par_rapport_x_gradM1_C_hab_sel", "der_sec_par_rapport_y_gradM1_C_hab_sel"),
    chunksize=500_000
):

    # ------------------------------------------------------------
    # FIRST: find the first timestep WITHOUT loading whole CSV
    # ------------------------------------------------------------
    first_timestep = None

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        first_timestep = chunk["TimeStep"].iloc[0]
        break

    print(f"Using first timestep: {first_timestep}")

    # ------------------------------------------------------------
    # SECOND: stream CSV and collect only this timestep
    # ------------------------------------------------------------
    df_list = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        df_list.append(chunk[chunk["TimeStep"] == first_timestep])

    df_t_long = pd.concat(df_list, ignore_index=True)
    df_t_long.columns = df_t_long.columns.str.strip()

    # ------------------------------------------------------------
    # Convert long → wide for this timestep only
    # ------------------------------------------------------------
    df_t = df_t_long.pivot_table(
        index=[y_col, x_col],
        columns="Field",
        values="Value",
        aggfunc="first"
    ).reset_index()

    # ------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    x_vals = np.sort(df_t[x_col].unique())
    y_vals = np.sort(df_t[y_col].unique())

    # ------------------------------------------------------------
    # Loop over variables
    # ------------------------------------------------------------
    for var in variables:

        if var not in df_t.columns:
            print(f"Variable '{var}' not found in timestep {first_timestep}.")
            continue

        grid = df_t.pivot_table(
            index=y_col,
            columns=x_col,
            values=var,
            aggfunc="first"
        )

        grid = grid.reindex(index=y_vals, columns=x_vals)
        Z = grid.values

        plt.figure(figsize=(6, 5))

        im = plt.imshow(
            Z,
            origin="lower",
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            cmap="viridis",
            aspect="equal"
        )

        plt.colorbar(im, label=var)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{var} (TimeStep={first_timestep})")

        output_path = os.path.join(output_dir, f"{var}_t{first_timestep}.png")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")
