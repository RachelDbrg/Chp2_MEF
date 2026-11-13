## Load the custom-functions
from convert_vtk_csv import vtu_to_csv
from heterogeneous_landscape import plot_maps_from_csv
from BC_overlap_all_pairs import plot_overlap_from_csv
from plot_overlaps_heatmap import plot_spatial_overlaps_grid
#from plots_2D_concentration import plot_spatial_snapshots
from memory_safe_2D_concentration import plot_spatial_snapshots
from plot_surface_through_time import plot_surface_through_time
from overlap_through_time import temporal_density_change
import os
from create_gif_through_time import create_species_gifs
from equilibrium_time import func_equilibrium_time
from test_3D_resources_plot import plot_gradient_heatmaps
from check_mass_conservation import compute_mass_and_plot



## Here, define the name of the files for which we want to produce figures
from config import current_simulation, simulation_folder, figures_folder, csv_name

simulation_data = os.path.join(simulation_folder, f"{current_simulation}")
print(simulation_data)

print(csv_name)


## Define the folder where to store the figures: figures/name_simulation
output_dir = os.path.join(figures_folder, current_simulation)
#print("output_dir for figures", output_dir)

## Create the subfolder for figures if it does not exist 
os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

## Convert the outputs into .csv
vtu_to_csv(simulation_data, csv_name)
#print("Done converting the files into .csv")

csv_path = os.path.join(simulation_folder, current_simulation, f"{csv_name}.csv")
print("Path to csv", csv_path)

## Check the mass conservation for all initially present species
compute_mass_and_plot(csv_path, output_dir)

## Plot the heterogeneous landscape over which the simulations are runs - land cover and heterogeneity
plot_maps_from_csv(csv_path, output_dir)
print("Done producing the landscape plot")
# --------------------------------------------------------------------------------------------------------------------------

## Produce the plot of spatial distribution
plot_spatial_snapshots(csv_path, output_dir)
print("Done producing the spatial snapshots")

## Produce the .gif of transient dynamics
#create_species_gifs(csv_path, output_dir)
print("Done producing gifs of transient dynamics")


## Produce the plot of the percentage of pairwise overlap through time, using BC index
#plot_overlap_from_csv(csv_path, output_dir)
print("Done producing the inter-species overlap plots")

## Plot the surface covered by the species through time 
#plot_surface_through_time(csv_path, output_dir)
print("Done producing the surface through time plots")


## Produce the spatial plot of overlap 
#plot_spatial_overlaps_grid(f"{simulation_data}.csv", output_dir)
#print("Done producing overlap heatmaps plots")

## Plot the temporal overal for each species
#temporal_density_change(csv_path, output_dir)
print("Done producing the intra-species temporal overlap plots")


## Time needed to each spatial stability
#eq_time = func_equilibrium_time(csv_path, output_dir)
#print(eq_time)

## 3D plots
#plot_gradient_heatmaps(csv_path, output_dir)
