import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def temporal_density_change(
    csv_path,
    output_dir,
    species_list=("C", "M", "P", "D1"),
    plot=True,
    output_name="overlap_through_time"
):

    df = pd.read_csv(csv_path)

    time_values = sorted(df["TimeStep"].unique())
    species = [sp for sp in species_list if sp in df.columns]

    results = []

    for sp in species:

        for i, t in enumerate(time_values):

            if i == 0:
                results.append({
                    "TimeStep": t,
                    "Species": sp,
                    "RelativeChange": 0.0
                })
                continue

            df_t = df[df["TimeStep"] == t]
            df_prev = df[df["TimeStep"] == time_values[i - 1]]

            # 🔒 align by space (IMPORTANT)
            df_merged = df_t.merge(df_prev, on=["x", "y"], suffixes=("_t", "_prev"))

            u = df_merged[f"{sp}_t"].values
            u_prev = df_merged[f"{sp}_prev"].values

            diff_norm = np.linalg.norm(u - u_prev)
            prev_norm = np.linalg.norm(u_prev)

            rel_change = diff_norm / prev_norm if prev_norm > 0 else 0

            results.append({
                "TimeStep": t,
                "Species": sp,
                "RelativeChange": rel_change
            })

    df_out = pd.DataFrame(results)

    print(df_out.head())
    print(df_out["RelativeChange"].describe())

    # -----------------------------
    # Plot
    # -----------------------------
    if plot:
        plt.figure(figsize=(8, 5))

        for sp in species:
            df_sp = df_out[df_out["Species"] == sp]
            plt.plot(df_sp["TimeStep"], df_sp["RelativeChange"], label=sp)

        plt.xlabel("Time")
        plt.ylabel("Relative density change")
        plt.title("Temporal change of species density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        #plt.show()

        output_path = os.path.join(output_dir, f"{output_name}.png")
        
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved plot → {output_path}")

    return df_out