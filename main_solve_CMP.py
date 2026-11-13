import sys
from mefpp4py import mefpp
import petsc4py as p4p
import numpy as np
import scipy as sp
import pandas as pd
import subprocess
from scipy.optimize import minimize
from pathlib import Path
import os
import re
import glob
import os
import shutil
from mpi4py import MPI
#import psutil
from fn_read_last_vtu import read_last_vtu
from def_conditions_init import setup_initial_conditions
from adimension_parms import setup_model_dependent_parameters


# =====================================================
# 1. Define the landscape used for the simulations
# =====================================================

## Get the current repo path
script_dir = Path(__file__).resolve().parent

# Choisit le .csv sur lequel sera fait les simulations
files_by_model = {
    "SM1": script_dir / "UD_monthly_caribou_wolf_mars_2022.csv",
    #"SM1": script_dir / "habitat_500m_normalized.csv",
    "SM2": script_dir / "UD_monthly_caribou_moose_avril_2023.csv",
}

## Defini un modele (pas vraiment utile si ci theoretical or debug)
## C'etait qque chose utile si on faisait du Nelder-Mead pour choisir le modele sur lequel on faisait les comparaisons
## utilise en mode empirical seulement
current_model = "SM1"

## Écrit le carreRachel-collection.champs correspondant, pour convertir le .csv en .mail
collection_file = os.path.join(
    os.path.dirname(__file__),
    "carreRachel-collection.champs"
)

with open(collection_file, "w") as fp:
    fp.write(
        f"convertisseurversgiref carreRachel "
        f"[{files_by_model[current_model].name}, "
        f"GRILLESOMMETCSV, 2, Lin]\n",
    )


# =====================================================
# 2. Define initial conditions
# =====================================================

## Load the empirical data
species_by_model = {    
    "SM1": ["caribou", "wolf"],
    "SM2": ["caribou", "moose"],
    "C"  : ["caribou"],
    "M"  : ["moose"],
    "P"  : ["wolf"],
    "D"  : ["deer"],
    "CM"  : ["caribou", "moose"],
    "CP"  : ["caribou", "wolf"],
    "CD"  : ["caribou", "deer"],
    "CMP"  : ["caribou", "moose", "wolf"],
    "CMPD"  : ["caribou", "moose", "wolf", "deer"]
    
}

## Ici on definit les CI que l'on veut  -- soit empirical soit theoretical, soit debug (pour verifier numeriquement nos équations)
ci_type = "theoretical"

## Ecrit le .champs avec les CI
(species,
    df,
    uds,
    initial_densities,
    positions,
    ci_description
) = setup_initial_conditions(
    ci_type=ci_type,
    species_by_model=species_by_model,
    files_by_model=files_by_model
)





# =====================================================
# 3. Define scenarios
# =====================================================

## Definition de tous les parametres que l'on peut faire varier 
all_param_groups = {
    "caribou": ["c1", "c2", "c3", "c4", "c5", "c6"],
    "moose":   ["m1", "m2", "m3", "m4", "m5", "m6"],
    "wolf":    ["p1", "p2", "p3", "p4", "p5", "p6"],
    "deer":    ["d1", "d2", "d3", "d4", "d5", "d6"],
}



## On attribue un budget global pour l'ensemble des mecanismes
c_IFD = "c1", "c2"
c_LOF = "c3", "c4", "c5", "c6"
m_IFD = "m1", "m2"
m_LOF = "m3", "m4", "m5", "m6"
d_IFD = "d1", "d2"
d_LOF = "d3", "d4", "d5", "d6"

parms_names = [
    p
    for params in all_param_groups.values()
    for p in params
]




# np.random.seed(22)
# theta_random = np.random.uniform(
#     0,
#     1,
#     size=(n_scenarios, len(parms_names))
# )

# # Met à 0 les paramètres des espèces absentes
# for i, p in enumerate(parms_names):

#     if p.startswith("c") and "caribou" not in species:
#         theta_random[:, i] = 0.0

#     elif p.startswith("m") and "moose" not in species:
#         theta_random[:, i] = 0.0

#     elif p.startswith("p") and "wolf" not in species:
#         theta_random[:, i] = 0.0

#     elif p.startswith("d") and "deer" not in species:
#         theta_random[:, i] = 0.0

# theta_scenarios = theta_random


# theta0 = np.zeros(len(parms_names))



# #####################
# def generate_species_parameters():

#     A_ifd = np.random.uniform(0, 3)
#     A_lof = np.random.uniform(0, 3)

#     ifd = (
#         A_ifd *
#         np.random.dirichlet(np.ones(2))
#     )

#     lof = (
#         A_lof *
#         np.random.dirichlet(np.ones(4))
#     )

#     return np.concatenate([ifd, lof])

# theta_random = np.zeros((n_scenarios, len(parms_names)))

# for k in range(n_scenarios):

#     col = 0

#     for species_name in [
#         "caribou",
#         "moose",
#         "wolf",
#         "deer"
#     ]:

#         if species_name in species:
#             theta_random[k, col:col+6] = (
#                 generate_species_parameters()
#             )
#         else:
#             theta_random[k, col:col+6] = 0.0

#         col += 6

# theta_scenarios = theta_random




# ######################

# ## Pour scenario i), tous les parms sont à 1 
# #theta_scenarios = np.ones((n_scenarios, len(parms_names)))

# # ## Pour le scenario ii), variations des parms sauf les 6 (proies naives)
# def generate_caribou_parameters():

#     A_ifd = np.random.uniform(0, 3)
#     A_lof = np.random.uniform(0, 3)

#     ifd = (
#         A_ifd *
#         np.random.dirichlet(np.ones(2))
#     )

#     lof = (
#         A_lof *
#         np.random.dirichlet(np.ones(3))
#     )

#     return np.concatenate([
#         ifd,    # c1,c2
#         lof,    # c3,c4,c5
#         [0.0]   # c6
#     ])

# def generate_moose_parameters():

#     A_ifd = np.random.uniform(0, 3)
#     A_lof = np.random.uniform(0, 3)

#     ifd = (
#         A_ifd *
#         np.random.dirichlet(np.ones(2))
#     )

#     lof = (
#         A_lof *
#         np.random.dirichlet(np.ones(3))
#     )

#     return np.concatenate([
#         ifd,    # m1,m2
#         lof,    # m3,m4,m5
#         [0.0]   # m6
#     ])

# theta_random = np.zeros((n_scenarios, len(parms_names)))

# for k in range(n_scenarios):

#     col = 0

#     for species_name in [
#         "caribou",
#         "moose",
#         "wolf",
#         "deer"
#     ]:

#         if species_name not in species:

#             theta_random[k, col:col+6] = 0.0

#         elif species_name == "caribou":

#             theta_random[k, col:col+6] = (
#                 generate_caribou_parameters()
#             )

#         elif species_name == "moose":

#             theta_random[k, col:col+6] = (
#                 generate_moose_parameters()
#             )

#         else:

#             theta_random[k, col:col+6] = (
#                 generate_species_parameters()
#             )

#         col += 6

# theta_scenarios = theta_random






## Pour le scenario iii), predateur a 1, parms qui donnent les meilleurs W pour C et M et 
## variation des parms de d

## Pour le scenario iv) et v) variation des parms de P + parms qui donnent les meilleurs W pour C et M
## et pas de D

## Pour le scenario iv) et v, parms qui donnent les meilleurs W pour M, C et D + variation de P 

######################

theta_file = os.path.join(
    os.path.dirname(__file__),
    "theta_master.csv"
)


df_theta = pd.read_csv(theta_file)

n_scenarios = len(df_theta)

theta_scenarios = (
    df_theta[parms_names]
    .to_numpy()
)

deer_cols = ["d1","d2","d3","d4","d5","d6"]

for col in deer_cols:
    theta_scenarios[:, parms_names.index(col)] = 0.0

theta0 = np.zeros(len(parms_names))


print(theta_scenarios)
print(parms_names)



values_scenario_file = os.path.join(
    os.path.dirname(__file__),
    "values_scenario.champs"
)



# Initialisation du fichier avant la boucle avec des 0 partouts
with open(values_scenario_file, "w") as f:

    f.write("## Initial scenario\n\n")

    for name, value in zip(parms_names, theta0):
        f.write(f"scalaire {name} {value}\n")


# =====================================================
# 4. Define output folders
# =====================================================

# ## Dossier dans lequel se trouvent les exports MEF
# export_dir = os.path.join(
#     os.path.dirname(__file__),
#     "simulations",
#     "ii"
# )

# ## Definition du dossier dans lequel on met les resultats
# results_root = os.path.join(
#     export_dir,
#     "CMP_"
# )

sim_root = os.path.join(
    os.path.dirname(__file__),
    "simulations",
    "ii"
)

export_dir = os.path.join(
    sim_root,
    "TMP_CMP"
)

results_root = os.path.join(
    sim_root,
    "CMP_"
)



os.makedirs(export_dir, exist_ok=True)
os.makedirs(results_root, exist_ok=True)
# os.makedirs(results_root, exist_ok=True)


# =====================================================
# 5. Obtain dimensionless parameters
# =====================================================

name_parm, value_parm, physical_parameters = (
    setup_model_dependent_parameters(current_model)
)

# Write dimensionless parameters in a .champs file
parms_adim_file = os.path.join(
    export_dir,
    "parms_adim.champs"
)

with open(parms_adim_file, "w") as f:

    f.write("## Dimensionless parameters\n\n")

    for name, value in zip(name_parm, value_parm):
        f.write(f"scalaire {name} {value}\n")

print(f"Dimensionless parameters written to: {parms_adim_file}")


# =====================================================
# 6. Define cleaning functions to remove files 
# inbetween simulations
# =====================================================

## Definition fonction de nettoyage 
# def clean_export_directory(export_dir):
#     """
#     Efface tous les fichiers produits par MEF++
#     avant le scénario suivant.
#     """
    
#     for pattern in [
#     "CMP_*.vtu",
#     "CMP_*.mail",
#     "CMP_*.pvtu",
#     "CMP_*.pvd",
#     "CMP_*.tex"
#     ]:
#         for f in glob.glob(
#             os.path.join(export_dir, pattern)
#         ):
#             try:
#                 os.remove(f)
#             except OSError as e:
#                 print(f"Impossible de supprimer {f}: {e}")
     

def clean_export_directory(export_dir):

    keep_files = {
        "parms_adim.champs", 
        "values_scenario.champs"
    }

    for name in os.listdir(export_dir):

        if name in keep_files:
            continue


        path = os.path.join(
            export_dir,
            name
        )

        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)

        elif os.path.isdir(path):
            shutil.rmtree(path)


# =====================================================
# 7. Loop the PDE solving over the scenarios
# =====================================================

mefpp.initialise("carreRachel") 
print("Initialisation du pb")

collection = mefpp.reqCollectionDeCorps()
collection.lisDonneesDeBase(["carreRachel"])
corps = collection.reqCorps("carreRachel")
gfc = corps.reqGFC()



comm = MPI.COMM_WORLD
rank = comm.Get_rank()



for scenario_id, theta in enumerate(theta_scenarios):

# Cas particulier pour skiper le 1er scenario (deja fait)
#for scenario_id, theta in enumerate(theta_scenarios[1:], start=1):

    pp_reinit = gfc.reqPP("pp_redemarre")
    pp_reinit.execute()

    
    # =====================================================
    # Nettoyage des exports précédents
    # =====================================================

    if rank == 0:

        clean_export_directory(
        export_dir
        )

        print(
            f"\n===== Scenario {scenario_id:04d} ====="
        )

        print(
            dict(zip(parms_names, theta))
        )

    # attendre que le nettoyage soit terminé (si on roule en //)
    comm.Barrier()

    # =====================================================
    # Paramètres du scénario
    # =====================================================

    theta = comm.bcast(theta, root=0)

    scenario_dir = os.path.join(
                results_root,
                f"scenario_{scenario_id:04d}"
            )
    
    os.makedirs(
                scenario_dir,
                exist_ok=True
            )



    values_scenario_file = os.path.join(
    export_dir,
    "values_scenario.champs"
)

    with open(values_scenario_file, "w") as f:

        for name, value in zip(parms_names, theta):
            f.write(f"scalaire {name} {value}\n")

    print(f"Scenario {scenario_id} written")


    ## Re-ecrit les champs selon le scenario actuel (hopefully)
    for name, value in zip(parms_names, theta):
        gfc.reqChamp(name).asgnValeur(value)

    # =====================================================
    # Résolution PDE
    # =====================================================

    gfc.executeActionsRecursif()

    # attendre que tous les ranks aient fini
    comm.Barrier()


    # =====================================================
    # Sauvegarde des résultats
    # =====================================================

    if rank == 0:

        # renomme les fichiers pour que les pas de temps soient tous les memes d'une simulation a l'autre
        
        # -------------------------------------------------
        # Renommer les VTU avec des indices réguliers
        # -------------------------------------------------

        vtu_files = sorted(
            glob.glob(
                os.path.join(
                    export_dir,
                    "*.cont.vtu"
                )
            ),
            key=os.path.getmtime
        )

        for i, old_file in enumerate(vtu_files):

            new_file = os.path.join(
                export_dir,
                f"snapshot_CMP_{i:04d}.cont.vtu"
            )

            os.rename(
                old_file,
                new_file
            )

        print(
            f"{len(vtu_files)} VTU files renamed"
        )


        # déplacer tous les exports MEF++
        for ext in [
            "*.vtu",
            "*.pvtu",
            "*.pvd",
            "*.tex"
        ]:

            for f in glob.glob(
                os.path.join(
                    export_dir,
                    ext
                )
            ):

                print("VTU trouvés :")
                print(glob.glob(os.path.join(export_dir, "*.vtu")))

                shutil.move(
                    f,
                    os.path.join(
                        scenario_dir,
                        os.path.basename(f)
                    )
                )

        for fname in [
            "parms_adim.champs",
            "values_scenario.champs"
        ]:
            src = os.path.join(export_dir, fname)

            if os.path.exists(src):
                shutil.copy2(
                    src,
                    os.path.join(
                        scenario_dir,
                        fname
                    )
                )

                

        # sauvegarde paramètres
        with open(
            os.path.join(
                scenario_dir,
                "parameters.txt"
            ),
            "w"
        ) as fp:

            fp.write(
                f"Scenario {scenario_id}\n\n"
            )

            fp.write(
                "=== Parametres utilises ===\n"
            )

            for name, value in zip(
                parms_names,
                theta
            ):

                fp.write(
                    f"{name} = {value}\n"
                )

            fp.write(
                "\n=== Parametres derives du modele ===\n"
            )

            for name, value in zip(
                name_parm,
                value_parm
            ):

                fp.write(
                    f"{name} = {value}\n"
                )

        print(
            f"Résultats sauvegardés dans {scenario_dir}"
        )

    # attendre que le déplacement des fichiers soit terminé
    comm.Barrier()

mefpp.finalise()