import sys
from mefpp4py import mefpp
import petsc4py as p4p
import numpy as np
import scipy as sp
import pandas as pd
import subprocess
from scipy.optimize import minimize
import os
import re
import glob
import os
import shutil
from mpi4py import MPI


## Ici on choisit le modèle que l'on veut tester
current_model = "CMP"


## Definition de tous les parametres que l'on peut faire varier 
param_groups = {
    "caribou": ["c1", "c2", "c3", "c4", "c5"],
    "moose": ["m1", "m2", "m3", "m4", "m5"],
    "wolf": ["p3", "p4", "p5"]
}

parms_names = []

for group in param_groups.values():
    parms_names.extend(group)

print(parms_names)


## Fait 10 combinaisons différentes de tous les scenarios
theta0 = np.array([
    1,  # c1
    1,  # c2
    1,  # c3
    1,  # c4
    1,  # c5
    1,  # m1
    1,  # m2
    1,  # m3
    1,  # m4
    1,  # m5
    1,  # p3
    1,  # p4
    1   # p5
])

np.random.seed(11)


## Meme avec supg, ca diverge parfois...
theta_scenarios = np.vstack([
    #np.zeros((1, 13)),
    #np.ones((1, 13)),
    np.random.uniform(0, 10, (10, 13))
])


print(theta_scenarios)



### 2. Write these parameters in a .champs file that will be loaded before solving the PDE
def write_parameters(gfc, theta):

    
    for name, val in zip(parms_names, theta):
        gfc.reqChamp(name).asgnValeur(val)

    print(
        "Parameters:",
        dict(zip(parms_names, theta))
    )



source_dir = os.path.join(
    os.path.dirname(__file__),
    "simulations",
    "vars_parms_c",
    current_model
)
os.makedirs(source_dir, exist_ok=True)


print(source_dir)



## Import here all the custom functions defined in other scripts
from convert_vtk_csv import vtu_to_csv
from heterogeneous_landscape import plot_maps_from_csv
from BC_overlap_all_pairs import plot_overlap_from_csv
from plot_overlaps_heatmap import plot_spatial_overlaps_grid
#from plots_2D_concentration import plot_spatial_snapshots
from memory_safe_2D_concentration import plot_spatial_snapshots
from plot_surface_through_time import plot_surface_through_time
from overlap_through_time import temporal_density_change
from create_gif_through_time import create_species_gifs
from equilibrium_time import func_equilibrium_time
from test_3D_resources_plot import plot_gradient_heatmaps
from check_mass_conservation import compute_mass_and_plot


## THIS IS THE WHOLE FUNCTION THAT ESTIMATES PARAMETERS, RUN THE PDE, COMPARE ITS OUTPUTS TO EMPIRICAL DATA
## ADJUST THE PARAMETERS AND TRIES TO MINIMIZE THE DISTANCE BETWEEN ESTIMATION AND OBSERVATION

### Load the config files, specifying the path and simulations parameters
#from config import current_simulation, simulation_folder

mefpp.initialise("carreRachel") 
print("Initialisation du pb")


collection = mefpp.reqCollectionDeCorps()
collection.lisDonneesDeBase(["carreRachel"])
corps = collection.reqCorps("carreRachel")
gfc = corps.reqGFC()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def objective(theta):

    if not hasattr(objective, "iter"):
        objective.iter = 0

    else: 
        objective.iter += 1

    print(
    f"Rank {rank} iter {objective.iter} "
    f"theta[0]={theta[0]}"
)

    write_parameters(gfc, theta)

    


    gfc.executeActionsRecursif()

    # Everyone finished the solve
    comm.Barrier()

    ## a chaque execution, MEF envoie les resultats (vtu du dernier pas de temps dans le dossier /simulations)

    # Dossier où MEF++ exporte les VTU
    export_dir = os.path.join(
    os.path.dirname(__file__),
    "simulations",    "vars_parms_c"
)

# Dossier de l'itération courante
    iter_dir = os.path.join(
    export_dir,
    current_model,
    f"iter_{objective.iter:04d}"
)

    os.makedirs(iter_dir, exist_ok=True)

    

    ## Si on roule en parallèle, il faut que ces actions ne soient faites que 1 fois sinon ca plante


    # Seul le rank 0 effectue les opérations sur le système de fichiers
    if rank == 0:

        # Déplacer tous les VTU du dossier simulations
        for f in glob.glob(os.path.join(export_dir, "*.vtu")):
            shutil.move(
                f,
                os.path.join(iter_dir, os.path.basename(f))
            )

        # Déplacer tous les PVD
        for f in glob.glob(os.path.join(export_dir, "*.pvd")):
            shutil.move(
                f,
                os.path.join(iter_dir, os.path.basename(f))
            )

        # Déplacer le pvtu pour reconstruire le maillage
        for f in glob.glob(os.path.join(export_dir, "*.pvtu")):
            shutil.move(
                f,
                os.path.join(iter_dir, os.path.basename(f))
            )
        # Sauvegarder les paramètres
        params_file = os.path.join(iter_dir, "parameters.txt")

        with open(params_file, "w") as fp:
            fp.write(f"Iteration: {objective.iter}\n\n")            
            for name, value in zip(parms_names, theta):
                fp.write(f"{name} = {value}\n")

        # Nettoyage du dossier source
        iter_pattern = re.compile(r"^iter_\d+$")

        for path in glob.glob(os.path.join(source_dir, "*")):
            basename = os.path.basename(path)

            if os.path.isdir(path) and iter_pattern.match(basename):
                continue

            if os.path.isfile(path):
                os.remove(path)

  
for i, theta in enumerate(theta_scenarios):

    print("\n")
    print("="*60)
    print(f"SCENARIO {i+1}")
    print("="*60)

    print(dict(zip(parms_names, theta)))

    objective(theta)


