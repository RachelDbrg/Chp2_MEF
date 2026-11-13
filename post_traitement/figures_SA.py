import os
from convert_vtk_csv import vtu_to_csv
from heterogeneous_landscape import plot_maps_from_csv
from memory_safe_2D_concentration import plot_spatial_snapshots
from check_mass_conservation import compute_mass_and_plot
from config import (simulation_folder,csv_name)

figures_folder = "figures/"

# Tous les sous-dossiers de simulations
simulations = sorted([
    d for d in os.listdir(simulation_folder)
    if os.path.isdir(
        os.path.join(simulation_folder, d)
    )
])

print("Found simulations:")
for sim in simulations:
    print(sim)

for simulation_name in simulations:

    print("\n" + "="*60)
    print(f"Processing {simulation_name}")
    print("="*60)

    simulation_data = os.path.join(
        simulation_folder,
        simulation_name
    )

    output_dir = os.path.join(
        figures_folder,
        simulation_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    try:

        # Conversion VTU -> CSV
        vtu_to_csv(
            simulation_data,
            csv_name
        )

        csv_path = os.path.join(
            simulation_data,
            f"{csv_name}.csv"
        )

        print("CSV:", csv_path)

        # Conservation de masse
        compute_mass_and_plot(
            csv_path,
            output_dir
        )

        # Carte du paysage
        plot_maps_from_csv(
            csv_path,
            output_dir
        )

        # Distributions spatiales
        plot_spatial_snapshots(
            csv_path,
            output_dir
        )

        print(
            f"Finished {simulation_name}"
        )

    except Exception as e:

        print(
            f"ERROR in {simulation_name}: {e}"
        )