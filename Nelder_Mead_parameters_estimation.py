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
import psutil
from fn_read_last_vtu import read_last_vtu


## Load the empirical data
species_by_model = {
    "SM1": ["caribou", "wolf"],
    "SM2": ["caribou", "moose"]
}

## Specifier les fichiers empiriques a utiliser selon le modele

from pathlib import Path

script_dir = Path(__file__).resolve().parent

files_by_model = {
    "SM1": script_dir / "UD_monthly_caribou_wolf_mars_2022.csv",
    "SM2": script_dir / "UD_caribou_moose_2023.csv",
}


## Ici on choisit le modèle que l'on veut tester
current_model = "SM1"


species = species_by_model[current_model]
data_file = files_by_model[current_model]

df = pd.read_csv(data_file)
df = df.sort_values(["x", "y"])

uds = {}

# for sp in species:
#     col = f"ud_{sp}"
#     uds[sp] = df[col].values
#     uds[sp] = uds[sp] / uds[sp].sum()

for sp in species:
    col = f"{sp}_CF"
    uds[sp] = df[col].values
    uds[sp] = uds[sp] / uds[sp].sum()


## Definition de tous les parametres que l'on peut faire varier 
param_groups = {
    "caribou": ["c1", "c2", "c3", "c4", "c5"],
    "moose": ["m1", "m2", "m3", "m4", "m5"],
    "wolf": ["p3", "p4", "p5"]
}

parms_names = []

## On ne fait varier que les parametres pour les especes presentes
for sp in species:
    parms_names.extend(param_groups[sp])

#theta0 = np.ones(len(parms_names))

theta0 = np.ones(len(parms_names))

print(theta0)


### 2. Write these parameters in a .champs file that will be loaded before solving the PDE
def write_parameters(gfc, theta):

    #PARAM_FILE = "/users/rdubourg/Documents/Fir_sep_2026/fitted_parameters.champs"

        #with open(PARAM_FILE, "w") as f:
    
    for name, val in zip(parms_names, theta):
        gfc.reqChamp(name).asgnValeur(val)
        #theta_parms += f"scalaire {name} {val}\n"

    print(
        "Parameters:",
        dict(zip(parms_names, theta))
    )

## Ajuster tous les parms qui dependent de la taille du paysage

if current_model == "SM1":

    #L = 404.45

    ## Valeur pour mars 2022
    L = 405.606

elif current_model == "SM2":

    L = 344.779


## Rayons biologiques

rayon_perception_C = 4.7
rayon_perception_M = 1.0
rayon_perception_P = 0.2
rayon_perception_D1 = 1.0


## Rayons normalisés

rayon_perception_C_norm = rayon_perception_C / L
rayon_perception_M_norm = rayon_perception_M / L
rayon_perception_P_norm = rayon_perception_P / L
rayon_perception_D1_norm = rayon_perception_D1 / L


name_parm = [
    "L",
    "rayon_perception_C_norm",
    "rayon_perception_M_norm",
    "rayon_perception_P_norm",
    "rayon_perception_D1_norm"
]

value_parm = [
    L,
    rayon_perception_C_norm,
    rayon_perception_M_norm,
    rayon_perception_P_norm,
    rayon_perception_D1_norm
    #0.001, 0.001, 0.001, 0.001,
    #L, 0.01, 0.01, 0.01, 0.01
]


def change_parms_model_dependant(gfc, value_parm):

    for var_nam, var_val in zip(name_parm, value_parm):

        gfc.reqChamp(var_nam).asgnValeur(var_val)

    print(
        "parametres dependant du modele:",
        dict(zip(name_parm, value_parm))
    )





source_dir  = os.path.join(os.path.dirname(__file__),"simulations", "Nelder_Mead", f"NM_{current_model}")
os.makedirs(source_dir, exist_ok=True)


print(source_dir)


### 3. At each iteration, write the set of parameters and the score of the loss function (ie the BC distance)

best_loss = np.inf

def save_best_parameters(theta, loss):

    global best_loss

    if loss < best_loss:

        best_loss = loss

        
        BEST_FILE = (
            f"/home/rdubourg/scratch/Chp2/Nelder_Mead_sep_2026/"
            f"best_parameters_{current_model}.champs"
        )

        with open(BEST_FILE, "w") as f:

            for name, value in zip(parms_names, theta):
                f.write(f"scalaire {name} {value}\n")

        print(
            f"New best loss: {loss:.6e}"
        )

## 4. Compute the BC distance, which compares the empirical UD with the PDE output

def bhattacharyya_distance(p, q):

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize in case inputs are not exactly probabilities
    p = p / p.sum()
    q = q / q.sum()

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))

    # Numerical protection
    bc = np.clip(bc, 1e-15, 1.0)

    # Bhattacharyya distance
    distance = -np.log(bc)

    print("Distance: ", distance)
    print("Distance: ", distance)

    return distance


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

change_parms_model_dependant(gfc, value_parm)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()



def objective(theta):

    if not hasattr(objective, "iter"):
        objective.iter = 0
    else:
        objective.iter += 1

    if rank == 0:
        print(
            f"Rank {rank} iter {objective.iter} "
            f"theta={theta}"
        )

    # -------------------------------------------------
    # Rank 0 reçoit theta de Nelder-Mead
    # puis le diffuse à tous les processus
    # -------------------------------------------------

    theta = comm.bcast(theta, root=0)

    # -------------------------------------------------
    # Tous les processus configurent leur GFC
    # -------------------------------------------------

    write_parameters(gfc, theta)

    # -------------------------------------------------
    # Tous les processus participent au calcul MEF++
    # -------------------------------------------------

    ## Verification de l'usage de la memoire avant
    process = psutil.Process(os.getpid())
    print(
    "RSS before solve:",
    process.memory_info().rss / 1024**3,
    "GB"
    )

    gfc.executeActionsRecursif()

    # Tout le monde doit avoir terminé
    comm.Barrier()

    ## Verification de l'usage de la memoire apres

    print(
    "RSS after solve:",
    process.memory_info().rss / 1024**3,
    "GB"
    )

    # -------------------------------------------------
    # À partir d'ici, seulement rank 0 manipule les fichiers
    # -------------------------------------------------

    if rank == 0:

        export_dir = os.path.join(
            os.path.dirname(__file__),
            "simulations/Nelder_Mead/"
        )

        iter_dir = os.path.join(
            export_dir,
            f"NM_{current_model}",
            f"iter_{objective.iter:04d}"
        )

        os.makedirs(iter_dir, exist_ok=True)

        # Déplacer les VTU
        for f in glob.glob(os.path.join(export_dir, "*.vtu")):
            shutil.move(
                f,
                os.path.join(iter_dir, os.path.basename(f))
            )

        # Déplacer les PVD
        for f in glob.glob(os.path.join(export_dir, "*.pvd")):
            shutil.move(
                f,
                os.path.join(iter_dir, os.path.basename(f))
            )

        # Déplacer les PVTU
        for f in glob.glob(os.path.join(export_dir, "*.pvtu")):
            shutil.move(
                f,
                os.path.join(iter_dir, os.path.basename(f))
            )

        # Sauvegarder les paramètres
        params_file = os.path.join(
            iter_dir,
            "parameters.txt"
        )

        with open(params_file, "w") as fp:

            fp.write(
                f"Iteration: {objective.iter}\n\n"
            )

            for name, value in zip(parms_names, theta):
                fp.write(
                    f"{name} = {value}\n"
                )

        # -------------------------------------------------
        # Calcul du loss
        # -------------------------------------------------


        print(
        "RSS before read_last_vtu:",
        process.memory_info().rss / 1024**3,
        "GB"
        )   

        pde_caribou, pde_wolf, pde_moose = \
        read_last_vtu(iter_dir, "para")


        pde_caribou, pde_wolf, pde_moose = \
            read_last_vtu(iter_dir, "para")

        print(
        "RSS after read_last_vtu:",
        process.memory_info().rss / 1024**3, "GB")

        pde_uds = {
            "caribou": pde_caribou,
            "wolf": pde_wolf,
            "moose": pde_moose
        }

        losses = {
            sp: bhattacharyya_distance(
                uds[sp],
                pde_uds[sp]
            )
            for sp in species
        }

        loss = sum(losses.values())

        print(
        "RSS after BC:",
        process.memory_info().rss / 1024**3, "GB")

        print(
            ", ".join(
                f"{sp}={losses[sp]:.6f}"
                for sp in species
            )
            + f", Total={loss:.6f}"
        )

        save_best_parameters(theta, loss)

    else:

        loss = None

    # -------------------------------------------------
    # Le loss est envoyé à tous les ranks
    # -------------------------------------------------

    loss = comm.bcast(loss, root=0)

    return loss


result = minimize(
    objective,
    theta0, ## Initial guess for the parameters values (the theta vector we change at each iteration)
    method="Nelder-Mead",
    options={
        "maxiter": 1,
        "disp": True
    }
)
