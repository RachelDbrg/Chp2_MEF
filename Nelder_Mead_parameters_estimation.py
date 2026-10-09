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
#import psutil
from fn_read_last_vtu import read_last_vtu



## Ici on choisit le modèle que l'on veut tester
current_model = "SM1"

## Ici on definit les CI que l'on veut  -- soit empirical soit theoretical, soit debug (pour verifier numeriquement nos équations)
ci_type = "debug"

## On precise si on test le modele IFD (ie habitat-selection driven), ou non
model_type = "IFD"
#model_type = "LOF"
#model_type = "" ## C en fait le defaut (IFD+LOF)


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
    "SM2": script_dir / "UD_monthly_caribou_moose_avril_2023.csv",
}


## Écrit le carreRachel-collection.champs correspondant, pour convertir le .csv en .mail
collection_file = os.path.join(
    os.path.dirname(__file__),
    "carreRachel-collection.champs"
)

with open(collection_file, "w") as fp:
    fp.write(
        f"convertisseurversgiref carreRachel "
        f"[{files_by_model[current_model].name}, "
        f"GRILLESOMMETCSV, 2, Lin]\n"
    )



## Distributions initiales des especes si empirique, ie provient .csv
empirical_ci = {
    "SM1": {
        "C": ("carreRachel.caribou_CI.cmefpp"),
        "P": "carreRachel.wolf_CI.cmefpp",
        "M": 0,
        "D1": 0,
    },

    "SM2": {
        "C": "carreRachel.caribou_CI.cmefpp",
        "P": 0,
        "M": "carreRachel.moose_CI.cmefpp",
        "D1": 0,
    }
}


## Definition de 10 positions aléatoires par espèce native lorsque ci_type is set to "theoretical"
import random

random.seed(69)

positions_C = {}
positions_M = {}
positions_P = {}


etalementCloche = 300

densiteInitC = 1
densiteInitM = 0
densiteInitP = 0
densiteInitD1 = 0 


for i in range(10):

    x = random.uniform(0.05, 0.95)
    y = random.uniform(0.05, 0.95)
    positions_C[f"C_{i+1}"] = f"((x-{x})^2 + (y-{y})^2)"

    x = random.uniform(0.05, 0.95)
    y = random.uniform(0.05, 0.95)
    positions_M[f"M_{i+1}"] = f"((x-{x})^2 + (y-{y})^2)"

    x = random.uniform(0.05, 0.95)
    y = random.uniform(0.05, 0.95)
    positions_P[f"P_{i+1}"] = f"((x-{x})^2 + (y-{y})^2)"

gaussian_sum_C = " + ".join(
    [f"exp((-etalementCloche*C_{i+1}))" for i in range(10)]
)

gaussian_sum_M = " + ".join(
    [f"exp((-etalementCloche*M_{i+1}))" for i in range(10)]
)

gaussian_sum_P = " + ".join(
    [f"exp((-etalementCloche*P_{i+1}))" for i in range(10)]
)

theoretical_ci = {
    "C": gaussian_sum_C,
    "M": gaussian_sum_M,
    "P": gaussian_sum_P
}


## On set toutes les distributions comme des gaussiennes au centre
debug_ci = {
    "C": "densiteInitC*exp(-etalementCloche*((x-0.5)^2 + (y-0.5)^2))",
    "M": "densiteInitM*exp(-etalementCloche*((x-0.5)^2 + (y-0.5)^2))",
    "P": "densiteInitP*exp(-etalementCloche*((x-0.5)^2 + (y-0.5)^2))",
    "D1": "densiteInitD1*exp(-etalementCloche*((x-0.5)^2 + (y-0.5)^2))"
    }


species = species_by_model[current_model]
data_file = files_by_model[current_model]


df = pd.read_csv(data_file)
df = df.sort_values(["x", "y"])

uds = {}

for sp in species:
    col = f"{sp}_CF"
    uds[sp] = df[col].values
    uds[sp] = uds[sp] / uds[sp].sum()



#Écrit le .champs qui correspond aux conditions initiales, selon le modèle
CI_FILE = os.path.join(
    os.path.dirname(__file__),
    "initial_conditions.champs"
)


with open(CI_FILE, "w") as fp:

    if ci_type == "empirical":

        for field_name, value in empirical_ci[current_model].items():

            if isinstance(value, str):

                fp.write(
                    f'scallin {field_name} "{value}"\n'
                )

            else:

                fp.write(
                    f"scallin {field_name} {value}\n"
                )


    elif ci_type == "theoretical":

        fp.write(f"scalaire etalementCloche {etalementCloche}\n")

        fp.write(f"scalaire densiteInitC {densiteInitC}\n")
        fp.write(f"scalaire densiteInitM {densiteInitM}\n")
        fp.write(f"scalaire densiteInitP {densiteInitP}\n")
        fp.write(f"scalaire densiteInitD1 {densiteInitD1}\n")

        for name, expr in positions_C.items():
            fp.write(f"scalaire {name} {expr}\n")

        for name, expr in positions_M.items():
            fp.write(f"scalaire {name} {expr}\n")

        for name, expr in positions_P.items():
            fp.write(f"scalaire {name} {expr}\n")

        fp.write(
            f"scalquad C densiteInitC*({theoretical_ci['C']})\n"
        )

        fp.write(
            f"scalquad M densiteInitM*({theoretical_ci['M']})\n"
        )

        fp.write(
            f"scalquad P densiteInitP*({theoretical_ci['P']})\n"
        )

        fp.write(
            "scalaire custom_D1 (y < 0.02)\n"
        )

        fp.write(
            "scalquad D1 densiteInitD1*custom_D1\n"
        )


    elif ci_type == "debug":

        fp.write(f"scalaire etalementCloche {etalementCloche}\n")

        fp.write(f"scalaire densiteInitC {densiteInitC}\n")
        fp.write(f"scalaire densiteInitM {densiteInitM}\n")
        fp.write(f"scalaire densiteInitP {densiteInitP}\n")
        fp.write(f"scalaire densiteInitD1 {densiteInitD1}\n")

        fp.write(
                    f"scalquad C densiteInitC*({debug_ci['C']})\n"
                )
        fp.write(
                            f"scalquad M densiteInitM*({debug_ci['M']})\n"
                        )
        fp.write(
                            f"scalquad P densiteInitP*({debug_ci['P']})\n"
                        )
        fp.write(
                            f"scalquad D1 densiteInitD1*({debug_ci['D1']})\n"
                        )
        

## Verifie si la somme de mes UD pour les CI, par espèces somment à 1
for sp in species:
    col = f"{sp}_CF"

    values = df[col].values

    print(
        f"{sp}: "
        f"min={values.min():.6e}, "
        f"max={values.max():.6e}, "
        f"sum={values.sum():.12f}"
    )



## Definition de tous les parametres que l'on peut faire varier 
param_groups = {
    "caribou": ["c1", "c2", "c3", "c4", "c5", "c6"],
    "moose":   ["m1", "m2", "m3", "m4", "m5", "m6"],
    "wolf":    ["p1", "p2", "p3", "p4", "p5", "p6"]
}

interaction_params = [
     "c3", "c4", "c5", "c6",
     "m3", "m4", "m5", "m6",
     "p1", "p2"
    ]

habitat_selection_params = [
    "c1", "c2",
    "m1", "m2",
    "p3", "p4", "p5", "p6"
    ]

parms_names = []

for sp in species:
    params = param_groups[sp]

    ## Ajoute une exception si D1 pas présente, pas besoin de chercher ses parms
    if "D1" not in species:
        params = [p for p in params if not p.endswith("6")]

    parms_names.extend(params)

theta0 = np.ones(len(parms_names))

## Si on est dans le modele IFD (habitat-selection driven),
## les seuls mécanismes qu'on regarde sont liés à l'habitat.
## Alors on force tous les paramètres d'interaction à 0.

if model_type == "IFD":
    theta0 = np.array([
        0.0 if p in interaction_params else 1
        for p in parms_names
    ])

elif model_type == "LOF":
    theta0 = np.array([
            0.0 if p in habitat_selection_params else 1.0
            for p in parms_names
        ])


print(parms_names)
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



## Ajuster tous les parms qui dependent de la taille du paysage -- en gros c'est l'etape de la normalisation 
## de mes coefficients selon la taille L de mon paysage

if current_model == "SM1":

    #L = 404.45

    ## Valeur pour mars 2022 - Nelder Mead
    L = 405.606

elif current_model == "SM2":

    L = 344.779


## Rayons biologiques
rayon_perception_C = 4.7
rayon_perception_M = 1.0
rayon_perception_P = 0.2
rayon_perception_D1 = 1.0

## Coefficients de diffusions empiriques (km2/an) (hp qu'ils sont constants)
coeff_diffusion_C =  1073.1
coeff_diffusion_M =  142.35 
coeff_diffusion_D1 = coeff_diffusion_M
coeff_diffusion_P  = 1051.25


## Rayons normalisés
rayon_perception_C_norm = rayon_perception_C / L
rayon_perception_M_norm = rayon_perception_M / L
rayon_perception_P_norm = rayon_perception_P / L
rayon_perception_D1_norm = rayon_perception_D1 / L


## Coefficients de diffusion normalisés, centrés sur C
coeff_diffusion_C_norm = 1
coeff_diffusion_M_norm = coeff_diffusion_M/coeff_diffusion_C
coeff_diffusion_P_norm = coeff_diffusion_P/coeff_diffusion_C
coeff_diffusion_D1_norm = coeff_diffusion_D1/coeff_diffusion_C


## Coefficient d'advection biologique pour WTD (km/year)
advection_rate_D1 = 5

## Coefficient d'advection normalisé (adimensionnel)
advection_rate_D1_norm = advection_rate_D1 * L / coeff_diffusion_C



name_parm = [
    "L",
    "rayon_perception_C_norm",
    "rayon_perception_M_norm",
    "rayon_perception_P_norm",
    "rayon_perception_D1_norm",
    "coeff_diffusion_C_norm",
    "coeff_diffusion_M_norm",
    "coeff_diffusion_P_norm",
    "coeff_diffusion_D1_norm",
    "advection_rate_D1_norm"
]

value_parm = [
    L,
    rayon_perception_C_norm,
    rayon_perception_M_norm,
    rayon_perception_P_norm,
    rayon_perception_D1_norm,
    coeff_diffusion_C_norm,
    coeff_diffusion_M_norm,
    coeff_diffusion_P_norm,
    coeff_diffusion_D1_norm,
    advection_rate_D1_norm
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
    # process = psutil.Process(os.getpid())
    # print(
    # "RSS before solve:",
    # process.memory_info().rss / 1024**3,
    # "GB"
    # )

    gfc.executeActionsRecursif()

    # Tout le monde doit avoir terminé
    comm.Barrier()

    ## Verification de l'usage de la memoire apres

    # print(
    # "RSS after solve:",
    # process.memory_info().rss / 1024**3,
    # "GB"
    # )

    
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

            fp.write(f"Iteration: {objective.iter}\n\n")
        
            fp.write("=== Parametres optimises ===\n")
            for name, value in zip(parms_names, theta):
                fp.write(f"{name} = {value}\n")

            ## Écrit les valeurs des paramètres adimensionnels
            fp.write("\n=== Parametres derives du modele ===\n")
            for name, value in zip(name_parm, value_parm):
                fp.write(f"{name} = {value}\n")

            ## Écrit les valeurs des paramètres physiques
            fp.write("\n=== Parametres physiques ===\n")
            fp.write(f"coeff_diffusion_C = {coeff_diffusion_C}\n")
            fp.write(f"coeff_diffusion_M = {coeff_diffusion_M}\n")
            fp.write(f"coeff_diffusion_P = {coeff_diffusion_P}\n")
            fp.write(f"coeff_diffusion_D1 = {coeff_diffusion_D1}\n")

            ## Écrit les densités initiales
            fp.write("\n=== Conditions initiales ===\n")
            fp.write(f"ci_type = {ci_type}\n")
            if ci_type == "empirical":

                for field, filename in empirical_ci[current_model].items():
                    fp.write(f"{field} = {filename}\n")

            elif ci_type == "theoretical":

                fp.write(f"scalaire etalementCloche {etalementCloche}\n")

                for name, expr in positions_C.items():
                    fp.write(f"scalaire {name} {expr}\n")

                for name, expr in positions_M.items():
                    fp.write(f"scalaire {name} {expr}\n")

                for name, expr in positions_P.items():
                    fp.write(f"scalaire {name} {expr}\n")

                fp.write(
                    f"scalquad C densiteInitC*({theoretical_ci['C']})\n"
                )

                fp.write(
                    f"scalquad M densiteInitM*({theoretical_ci['M']})\n"
                )

                fp.write(
                    f"scalquad P densiteInitP*({theoretical_ci['P']})\n"
                )

                fp.write(
                    "scalaire custom_D1 (y < 0.02)\n"
                )

                fp.write(
                    "scalquad D1 densiteInitD1*custom_D1\n"
                )


                

        # -------------------------------------------------
        # Calcul du loss
        # -------------------------------------------------


        # print(
        # "RSS before read_last_vtu:",
        # process.memory_info().rss / 1024**3,
        # "GB"
        # )   

        pde_caribou, pde_wolf, pde_moose = \
        read_last_vtu(iter_dir, "seq")

        # print(
        # "RSS after read_last_vtu:",
        # process.memory_info().rss / 1024**3, "GB")

        pde_uds = {
            "caribou": pde_caribou,
            "wolf": pde_wolf,
            "moose": pde_moose
        }

    ## Verifie si la somme de mes UD pour les CI, par espèces somment à 1
        for sp in species:
    
            pde = pde_uds[sp]
        
            print(
                f"{sp}: "
                f"min={pde.min():.6e}, "
                f"max={pde.max():.6e}, "
                f"sum={pde.sum():.12f}"
            )




        losses = {
            sp: bhattacharyya_distance(
                uds[sp],
                pde_uds[sp]
            )
            for sp in species
        }

        loss = sum(losses.values())

        # print(
        # "RSS after BC:",
        # process.memory_info().rss / 1024**3, "GB")

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
