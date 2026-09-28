import os
import glob
import re
from convert_vtk_csv import progress_bar, vtu_to_csv
from fn_read_last_vtu import read_vtu

## Extraire carte des ressources
from config import current_simulation, simulation_folder, csv_name

simulation_data = os.path.join(simulation_folder, f"{current_simulation}")

vtu_to_csv(simulation_data, csv_name)


## Renomme les vtu de simulations avec pas de temps locaux
from renomme_iterations_solv_inst_boucle import get_scenario_vtus

source_dir = "simulations/vars_parms_c/CMP"

scenario_vtus = get_scenario_vtus(source_dir)

for scenario, files in scenario_vtus.items():

    print(scenario)

    for f in files[:5]:
        print(os.path.basename(f))


## A partir du .csv, faire une carte de ressource, par espece ou les valeurs sont 1 si ressources profitable et 0 sinon
import pandas as pd

csv_file = os.path.join(
    simulation_data,
    f"{csv_name}.csv"
)

df = pd.read_csv(csv_file)

time_ref = df["TimeStep"].min()

## Recuper les champs de valeurs de paysage
landscape_df = (
    df[
        (df["Field"] == "valeurs_paysage")
        & (df["TimeStep"] == time_ref)
    ]
    .copy()
)

## Conversion de ce raster en champs de ressources profitables ou non 
## Habitats favorables pour le caribou: bryoids, conifers (open, dense, sparse)
good_habitats_caribou = [1, 2, 3, 4]

landscape_df["resource_caribou"] = (
    landscape_df["Value"]
    .isin(good_habitats_caribou)
).astype(int)

## Habitats favorables pour le moode: deciduous, herb, mixedwood, open, shrubs 
good_habitats_moose = [5, 6, 7, 8, 9]

landscape_df["resource_moose"] = (
    landscape_df["Value"]
    .isin(good_habitats_moose)
).astype(int)


print(
    landscape_df[
        ["Value",
         "resource_caribou",
         "resource_moose"]
    ].head()
)

landscape_sorted = landscape_df.sort_values(["x", "y"])

resource_caribou = (
    landscape_sorted["resource_caribou"]
    .values
)

resource_moose = (
    landscape_sorted["resource_moose"]
    .values
)

resource_caribou = (
    resource_caribou /
    resource_caribou.sum()
)

resource_moose = (
    resource_moose /
    resource_moose.sum()
)

env_df = pd.DataFrame({

    "x": landscape_sorted["x"].values,
    "y": landscape_sorted["y"].values,

    "resource_caribou": resource_caribou,
    "resource_moose": resource_moose

})

env_df.to_csv(
    "resources_caribou_moose.csv",
    index=False
)


resource_caribou = (
    landscape_df
    .sort_values(["x","y"])
    ["resource_caribou"]
    .values
)

resource_moose = (
    landscape_df
    .sort_values(["x","y"])
    ["resource_moose"]
    .values
)

resource_caribou = (
    resource_caribou /
    resource_caribou.sum()
)

resource_moose = (
    resource_moose /
    resource_moose.sum()
)

## Exporter les couches d'environnement binaires pour les deux espèces de proie
env_df = pd.DataFrame({
    "x": landscape_sorted["x"].values,
    "y": landscape_sorted["y"].values,
    "resource_caribou": resource_caribou,
    "resource_moose": resource_moose
    })

env_df.to_csv(
"figures/resources_caribou_moose.csv",
index=False
)



## Recuperer tous les .vtu dans le dossier des simulations
source_dir = os.path.join(
    "simulations",
    "vars_parms_c",
    current_simulation)

iter_dirs = sorted([
    os.path.join(source_dir, d)
    for d in os.listdir(source_dir)
    if os.path.isdir(os.path.join(source_dir, d))
    and d.startswith("iter")
])

scenario_vtus = {}

for iter_dir in iter_dirs:

    files = glob.glob(
        os.path.join(iter_dir, "*cont.vtu")
    )

    files.sort(
        key=lambda f: int(
            re.findall(
                r"\d+",
                os.path.basename(f)
            )[-1]
        )
    )

    scenario_vtus[
        os.path.basename(iter_dir)
    ] = files

for scen, files in scenario_vtus.items():

    print("\n", scen)

    for f in files[:10]:
        print(os.path.basename(f))


import matplotlib.pyplot as plt
import numpy as np
import os
import re

# -----------------------
# Bhattacharyya overlap
# -----------------------

def BA(p, q):

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)


    ## Force un minimum a zero lorsque les valeurs deviennent negatives (artefact numerique)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)

    p = p / p.sum()
    q = q / q.sum()

    return np.sum(np.sqrt(p * q))


# -----------------------
# Benefit / Risk score
# -----------------------

def J(prey, predator, resource, eps=1e-10):

    overlap_resource = BA(prey, resource)

    overlap_predator = BA(prey, predator)


    #print(overlap_resource)
    #print(overlap_predator)

    #return overlap_resource / (overlap_predator + eps)
    return  overlap_resource - overlap_predator


# -----------------------
# Plot J through time
# -----------------------

for scenario, files in scenario_vtus.items():

    times = []
    J_caribou_time = []
    J_moose_time = []

    for time, vtu_file in enumerate(files):

        time = int(
            re.search(
                r'_(\d+)\.cont\.vtu$',
                os.path.basename(vtu_file)
            ).group(1)
        )

        # ta fonction de lecture VTU
        densities = read_vtu(vtu_file, "para")

        caribou = densities["C"]
        moose   = densities["M"]
        wolf    = densities["P"]

        J_caribou_time.append(
            J(
                prey=caribou,
                predator=wolf,
                resource=resource_caribou
            )
        )

        J_moose_time.append(
            J(
                prey=moose,
                predator=wolf,
                resource=resource_moose
            )
        )

        times.append(time)

    output_dir = "plots_J"

    os.makedirs(output_dir, exist_ok=True)




    plt.figure(figsize=(8, 5))

    plt.plot(
        times,
        J_caribou_time,
        label="Caribou"
    )

    plt.plot(
        times,
        J_moose_time,
        label="Moose"
    )

    plt.axhline(
        1,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Time")

    plt.ylabel(
        "J = BA(resource, prey) / BA(prey, predator)"
    )

    plt.title(scenario)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{scenario}_J_through_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

## Test de figure avec toutes les iterations et tous les scenarios
plt.figure(figsize=(12, 8))

colors = {
    "Caribou": "green",
    "Moose": "red"
}

for scenario, files in scenario_vtus.items():

    files = sorted(files)

    times = []
    J_caribou_time = []
    J_moose_time = []

    for time, vtu_file in enumerate(files):

        densities = read_vtu(vtu_file, "para")

        caribou = densities["C"]
        moose   = densities["M"]
        wolf    = densities["P"]

        J_caribou_time.append(
            J(
                prey=caribou,
                predator=wolf,
                resource=resource_caribou
            )
        )

        J_moose_time.append(
            J(
                prey=moose,
                predator=wolf,
                resource=resource_moose
            )
        )

        times.append(time)

    if len(J_caribou_time) > 0:

        plt.plot(
            times[:len(J_caribou_time)],
            J_caribou_time,
            color="blue",
            alpha=0.8,
            label=f"{scenario} - Caribou"
        )

    if len(J_moose_time) > 0:

        plt.plot(
            times[:len(J_moose_time)],
            J_moose_time,
            color="red",
            alpha=0.8,
            label=f"{scenario} - Moose"
        )


## on definit une ligne horizontale a 0, car J = 0 
plt.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Time")
plt.ylabel(
    "J = BA(resource, prey) / BA(predator, prey")