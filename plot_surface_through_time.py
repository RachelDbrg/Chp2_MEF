import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_surface_through_time(
    csv_path,
    output_dir,
    species_list=("C", "M", "P", "D1"),
    output_name="surface_covered"
):
    """
    Create a plot of the surface of the domain covered by each species through time.

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
    active_species = [
        sp for sp in species_list
        if sp in df.columns and not np.allclose(df[sp], 0)
    ]

    if not active_species:
        print("No active species found.")
        return

    print("Active species:", active_species)

        # -----------------------------
    # Compute surface covered
    # -----------------------------
    surface_data = {sp: [] for sp in active_species}
    times = []

    # Compute the number of summits of the grid, once at initial time step
    total_cells = len(df[df["TimeStep"] == time_values_all[0]])
    print("n sommets=", total_cells)

    for t in time_values_all:
        ## Subset the .csv for each time step
        df_t = df[df["TimeStep"] == t]

        ## Check consistency in the size of the grid
        if len(df_t) != total_cells:
            raise ValueError(f"Inconsistent number of cells at timestep {t}")

        times.append(t)

        for sp in active_species:
            # Count cells where species is present
            ## The threshold value is super important!  
            covered = np.sum((df_t[sp].notna()) & (df_t[sp] > 1e-01))
            surface_fraction = covered / total_cells
            surface_data[sp].append(surface_fraction)




    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for sp in active_species:
        plt.plot(times, surface_data[sp], label=sp)

    plt.xlabel("Time step")
    plt.ylabel("Surface fraction covered")
    plt.title("Surface covered by species through time")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # Save figure
    # -----------------------------
    
    output_path = os.path.join(output_dir, f"{output_name}.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


    print(f"Figure saved to: {output_dir}")