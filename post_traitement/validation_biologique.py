import os
import re
import glob

import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vtk.util.numpy_support import vtk_to_numpy
from matplotlib.lines import Line2D


# ============================================================
# PARAMETRES
# ============================================================

species_names = [
    "C",
    "M",
    "P",
    "D"
]

species_labels = {
    "C": "Caribou",
    "M": "Moose",
    "P": "Wolf",
    "D": "Deer"
}

## Modele couple sur 100 pas de temps
root_dir = "/users/rdubourg/Documents/Chp2_MEF/simulations/i/CMPD_/scenario_0000"

## Modele decouple (sur 10 pas de temps seulement)
#root_dir = "/users/rdubourg/Documents/Chp2_MEF/simulations/validation_bio/CMPD_/scenario_0000"


## Est ce qu'on regarde un modèle couplé?
MODEL_COUPLED = True


# ============================================================
# UTILITAIRES
# ============================================================

def extract_time(filename):

    m = re.search(
        r"_(\d+)\.cont\.vtu$",
        filename
    )

    t = int(m.group(1))

    if "CMP_" in filename:
        t -= 101

    return t


# ============================================================
# LECTURE VTU
# ============================================================

def read_vtu(filename):

    reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(filename)

    reader.Update()

    return reader.GetOutput()


def point_to_cell(mesh):

    filt = vtk.vtkPointDataToCellData()

    filt.SetInputData(mesh)

    filt.Update()

    return filt.GetOutput()


def compute_areas(mesh):

    quality = vtk.vtkCellSizeFilter()

    quality.SetInputData(mesh)

    quality.ComputeAreaOn()

    quality.Update()

    out = quality.GetOutput()

    return vtk_to_numpy(
        out.GetCellData().GetArray("Area")
    )


def get_species_from_vtu(filename):

    mesh = read_vtu(filename)

    mesh = point_to_cell(mesh)

    areas = compute_areas(mesh)

    cell_data = mesh.GetCellData()

    data = {}

    for sp in species_names:

        data[sp] = vtk_to_numpy(
            cell_data.GetArray(sp)
        )

    return data, areas


def total_mass(field, areas):

    return np.sum(
        field * areas
    )


# ============================================================
# LISTE DES SCENARIOS
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            root_dir,
            "*.cont.vtu"
        )
    ),
    key=extract_time
)

print("Nombre de fichiers :", len(files))
print(files[:5])



# ============================================================
# VALIDATION BIOLOGIQUE
# ============================================================

def get_cell_arrays(filename):

    mesh = read_vtu(filename)
    mesh = point_to_cell(mesh)

    cell_data = mesh.GetCellData()

    arrays = {}

    for i in range(cell_data.GetNumberOfArrays()):

        name = cell_data.GetArrayName(i)

        arrays[name] = vtk_to_numpy(
            cell_data.GetArray(i)
        )

    return arrays


def interaction_response(
    focal,
    partner,
    areas
):
    available = np.sum(
        partner * areas
    ) / np.sum(areas)

    used = np.sum(
        focal * partner * areas
    ) / np.sum(
        focal * areas
    )

    return used / available


def directional_selection_ratio(
    focal,
    partner,
    areas
):

    partner_available = (
        np.sum(partner * areas)
        /
        np.sum(areas)
    )

    partner_used = (
        np.sum(
            focal * partner * areas
        )
        /
        np.sum(
            focal * areas
        )
    )

    return (
        (partner_used - partner_available) / partner_available
    )



# # ============================================================
# # LECTURE DES BETAS DANS LES .champs
# # ============================================================

def read_single_beta(
    filename
):

    with open(filename) as f:

        for line in f:

            if line.startswith("scalaire"):

                fields = line.split()

                return float(fields[2])

    return np.nan



def read_betas(filename, prefix):

    betas = {}

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if not line.startswith("scalaire"):
                continue

            tokens = line.split()

            if len(tokens) != 3:
                continue

            name = tokens[1]

            if not name.startswith(prefix):
                continue

            betas[name] = float(tokens[2])

    return betas


final_file = files[-1]


mesh = read_vtu(final_file)
mesh = point_to_cell(mesh)

areas = compute_areas(mesh)

arrays = {}

cell_data = mesh.GetCellData()

for i in range(cell_data.GetNumberOfArrays()):

    name = cell_data.GetArrayName(i)

    arrays[name] = vtk_to_numpy(
        cell_data.GetArray(i)
    )

D = arrays["M"]

print(sorted(arrays.keys()))



# # ============================================================
# # INDICE DE SELECTION DES HABITATS
# # ============================================================

def compute_selection_index(
    density,
    covariate,
    areas,
    betas,
    prefix
):

    total_mass = np.sum(
        density * areas
    )

    total_area = np.sum(
        areas
    )

    rows = []

    for coeff_name, beta in betas.items():

        class_id = int(
            coeff_name.replace(
                prefix,
                ""
            )
        )

        mask = (
            covariate == class_id
        )

        if np.sum(mask) == 0:
            continue

        class_mass = np.sum(
            density[mask]
            * areas[mask]
        )

        class_area = np.sum(
            areas[mask]
        )

        pop_fraction = (
            class_mass
            / total_mass
        )

        area_fraction = (
            class_area
            / total_area
        )

        selection_index = (
            pop_fraction
            / area_fraction
        )

        rows.append({
            "class_id": class_id,
            "beta": beta,
            "selection_index": selection_index,
            "population_fraction": pop_fraction,
            "area_fraction": area_fraction
            })

    return pd.DataFrame(rows)



from scipy.stats import spearmanr

def validate_selection(
    density,
    species_name,
    covariate,
    areas,
    beta_file,
    prefix,
    title=None
):

    betas = read_betas(
        beta_file,
        prefix
    )

    df = compute_selection_index(
        density=density,
        covariate=covariate,
        areas=areas,
        betas=betas,
        prefix=prefix
    )

    df = df.sort_values("beta")

    # ==================================
    # Statistiques
    # ==================================

    positive = df[
        df["beta"] > 0
    ]["selection_index"]

    negative = df[
        df["beta"] < 0
    ]["selection_index"]

    rho, p = spearmanr(
        df["beta"],
        df["selection_index"]
    )

    mean_beta_used = np.sum(
        df["beta"]
        * df["population_fraction"]
    )

    mean_beta_available = np.sum(
        df["beta"]
        * df["area_fraction"]
    )

    print("\n========================")
    print(species_name)
    print("========================")

    print(
        "Moyenne habitats attractifs :",
        positive.mean()
    )

    print(
        "Moyenne habitats répulsifs :",
        negative.mean()
    )

    print(
        "Différence :",
        positive.mean()
        - negative.mean()
    )

    print(
        f"Spearman rho = {rho:.3f}"
    )

    print(
        f"p-value = {p:.3e}"
    )

    print(
        "Beta moyen disponible :",
        mean_beta_available
    )

    print(
        "Beta moyen utilisé :",
        mean_beta_used
    )

    return {
        "species": species_name,
        "rho": rho,
        "p": p,
        "mean_attractive":
            positive.mean(),
        "mean_repulsive":
            negative.mean(),
        "beta_available":
            mean_beta_available,
        "beta_used":
            mean_beta_used
    }



## Validation sets
validation_sets = {

    "Habitat": {
        "covariate": arrays["valeurs_paysage"],
        "xlabel": r"Landcover selection coefficient ($\beta$)",

        "species_info": {

            "C": {
                "density": arrays["C"],
                "beta_file":
                    "/users/rdubourg/Documents/Chp2_MEF/"
                    "beta_Boreal_Caribou_habitat_selection_coefficients.champs",
                "prefix": "C_landcover",
                "title": "Caribou"
            },

            "M": {
                "density": arrays["M"],
                "beta_file":
                    "/users/rdubourg/Documents/Chp2_MEF/"
                    "beta_Moose_habitat_selection_coefficients.champs",
                "prefix": "M_landcover",
                "title": "Moose"
            },

            "D": {
                "density": arrays["D"],
                "beta_file":
                    "/users/rdubourg/Documents/Chp2_MEF/"
                    "beta_Deer_habitat_selection_coefficients.champs",
                "prefix": "D_landcover",
                "title": "Deer"
            }
        }
    },

    "Diversite": {
        "covariate": arrays["valeurs_diversity"],
        "xlabel": r"Diversity selection coefficient ($\beta$)",

        "species_info": {

            "C": {
                "density": arrays["C"],
                "beta_file":
                    "/users/rdubourg/Documents/Chp2_MEF/"
                    "beta_Boreal_Caribou_diversity_selection_coefficients.champs",
                "prefix": "C_diversity",
                "title": "Caribou"
            },

            "M": {
                "density": arrays["M"],
                "beta_file":
                    "/users/rdubourg/Documents/Chp2_MEF/"
                    "beta_Moose_diversity_selection_coefficients.champs",
                "prefix": "M_diversity",
                "title": "Moose"
            },

            "D": {
                "density": arrays["D"],
                "beta_file":
                    "/users/rdubourg/Documents/Chp2_MEF/"
                    "beta_Deer_diversity_selection_coefficients.champs",
                "prefix": "D_diversity",
                "title": "Deer"
            }
        }
    }
}


global_summary = []

for validation_name, params in validation_sets.items():

    print("\n")
    print("=" * 70)
    print(validation_name)
    print("=" * 70)

    species_info = params["species_info"]
    covariate = params["covariate"]

    results = []

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15,5)
    )

    for ax, (sp, info) in zip(
        axes,
        species_info.items()
    ):

        betas = read_betas(
            info["beta_file"],
            info["prefix"]
        )

        df = compute_selection_index(
            density=info["density"],
            covariate=covariate,
            areas=areas,
            betas=betas,
            prefix=info["prefix"]
        )

        df = df.sort_values("beta")

        rho, p = spearmanr(
            df["beta"],
            df["selection_index"]
        )

        slope = np.polyfit(
            df["beta"],
            df["selection_index"],
            1
        )[0]

        global_summary.append({

            "validation": validation_name,
            "species": sp,
            "rho": rho,
            "slope": slope,
            "p": p

        })

        colors = [
            "green" if b > 0
            else "red"
            for b in df["beta"]
        ]

        ax.scatter(
            df["beta"],
            df["selection_index"],
            c=colors,
            s=80,
            edgecolor="black"
        )

        xfit = np.linspace(
            df["beta"].min(),
            df["beta"].max(),
            100
        )

        ax.plot(
            xfit,
            np.polyval(
                np.polyfit(
                    df["beta"],
                    df["selection_index"],
                    1
                ),
                xfit
            ),
            color="black"
        )

        ax.axhline(
            1,
            ls="--",
            color="grey"
        )

        ax.axvline(
            0,
            ls=":",
            color="grey"
        )

        ax.text(
            0.05,
            0.95,
            (
                f"$\\rho$={rho:.2f}\n"
                f"p={p:.2e}\n"
                f"slope={slope:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8
            )
        )

        ax.set_title(info["title"])

        ax.set_xlabel(
            params["xlabel"]
        )

        ax.set_ylabel(
            "Selection index"
        )

    plt.tight_layout()
    plt.show()


## Donne un indice de co-occurence spatiale

if not MODEL_COUPLED:
    interaction_files = [
        (
            "P",
            "C",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Gray_Wolf_caribou_selection_coefficients.champs"
        ),
        (
            "P",
            "M",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Gray_Wolf_moose_selection_coefficients.champs"
        ),
        (
            "P",
            "D",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Gray_Wolf_deer_selection_coefficients.champs"
        )
    ]


else:
    interaction_files = [
        # -------------------------
        # Prédateur -> proies
        # -------------------------
        (
            "P",
            "C",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Gray_Wolf_caribou_selection_coefficients.champs"
        ),
        (
            "P",
            "M",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Gray_Wolf_moose_selection_coefficients.champs"
        ),
        (
            "P",
            "D",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Gray_Wolf_deer_selection_coefficients.champs"
        ),
        # -------------------------
        # Réponse des proies au loup
        # -------------------------
        (
            "C",
            "P",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Boreal_Caribou_wolf_selection_coefficients.champs"
        ),
        (
            "M",
            "P",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Moose_wolf_selection_coefficients.champs"
        ),
        (
            "D",
            "P",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Deer_wolf_selection_coefficients.champs"
        ),

        # -------------------------
        # Réponse des proies aux autres proies
        # -------------------------
        (
            "C",
            "C",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Boreal_Caribou_conspecific_selection_coefficients.champs"
        ),
        (
            "M",
            "M",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Moose_conspecific_selection_coefficients.champs"
        ),
        (
            "D",
            "D",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Deer_conspecific_selection_coefficients.champs"
        ),





        (
            "C",
            "M",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Boreal_Caribou_moose_selection_coefficients.champs"
        ),
        (
            "M",
            "C",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Moose_other_prey_selection_coefficients.champs"
        ),
        (
            "D",
            "M",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Deer_moose_selection_coefficients.champs"
        ),
        (
            "M",
            "D",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Moose_deer_selection_coefficients.champs"
        ),
        (
            "D",
            "C",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Deer_caribou_selection_coefficients.champs"
        ),
        (
            "C",
            "D",
            "/users/rdubourg/Documents/Chp2_MEF/"
            "beta_Boreal_Caribou_deer_selection_coefficients.champs"
        ),

    ]


rows = []

for focal, partner, filename in interaction_files:

    beta = read_single_beta(filename)

    overlap = interaction_response(
        focal=arrays[focal],
        partner=arrays[partner],
        areas=areas
    )

    directional = directional_selection_ratio(
        focal=arrays[focal],
        partner=arrays[partner],
        areas=areas
    )

    agreement = (
        np.sign(beta)
        ==
        np.sign(directional)
    )

    rows.append({

        "interaction":
            f"{focal}-{partner}",

        "beta":
            beta,

        "overlap":
            overlap,

        "directional":
            directional,

        "agreement":
            agreement

    })


df_inter = pd.DataFrame(rows)

print(df_inter)

fig, ax = plt.subplots(
    figsize=(6,5)
)

colors = [

    "green" if row["agreement"]
    else "red"

    for _, row in df_inter.iterrows()

]

marker_map = {

    # Prédateur -> proie
    "P-C": "^",
    "P-M": "^",
    "P-D": "^",

    # Proie -> prédateur
    "C-P": "v",
    "M-P": "v",
    "D-P": "v",

    # Conspecifiques
    "C-C": "o",
    "M-M": "o",
    "D-D": "o",

    # Herbivore -> herbivore
    "C-M": "s",
    "M-C": "s",
    "C-D": "s",
    "D-C": "s",
    "M-D": "s",
    "D-M": "s",
}


# On decale les etiquettes pour M-D et D-M
label_offsets = {

    "M-D": (8, 8),
    "D-M": (8, -12),

}



for _, row in df_inter.iterrows():

    ax.scatter(
        row["beta"],
        row["directional"],
        marker=marker_map[row["interaction"]],
        s=100,
        color=("green" if row["agreement"] else "red"),
        edgecolor="black"
    )

    offset = label_offsets.get(
        row["interaction"],
        (5, 5)
    )

    ax.annotate(
        row["interaction"],
        (row["beta"], row["directional"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=10
    )


ax.axhline(
    0,
    ls="--",
    color="grey"
)

ax.axvline(
    0,
    ls=":",
    color="grey"
)

ax.set_xlabel(
    r"Species interaction selection coefficient ($\beta$)"
)

ax.set_ylabel(
    "Relative directional effect"
)

# ax.set_title(
#     "Observed vs expected interaction responses"
# )

legend_elements = [

    # Prédateur -> proie
    Line2D(
        [0], [0],
        marker='^',
        color='w',
        markerfacecolor='lightgrey',
        markeredgecolor='black',
        markersize=8,
        label='Predator → Prey'
    ),

    # Proie -> prédateur
    Line2D(
        [0], [0],
        marker='s',
        color='w',
        markerfacecolor='lightgrey',
        markeredgecolor='black',
        markersize=8,
        label='Prey → Predator'
    ),

  Line2D(
    [0], [0],
    marker='s',
    color='w',
    markerfacecolor='lightgrey',
    markeredgecolor='black',
    markersize=8,
    label='Herbivore ⟷ Herbivore'
),

    # Conspecifiques
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor='lightgrey',
        markeredgecolor='black',
        markersize=8,
        label='Conspecific interaction'
    ),

    # Accord avec β
    Line2D(
        [0], [0],
        marker='o',
        color='green',
        linestyle='None',
        markersize=8,
        label='Consistent with β'
    ),

    # Désaccord avec β
    Line2D(
        [0], [0],
        marker='o',
        color='red',
        linestyle='None',
        markersize=8,
        label='Inconsistent with β'
    )

]

ax.legend(
    handles=legend_elements,
    loc="best"
)


plt.tight_layout()
plt.show()



summary = pd.DataFrame(
    global_summary
)



print(
    df_inter[
        ["interaction",
         "beta",
         "directional"]
    ]
)


print("\n")
print("=" * 70)
print("VALIDATION BIOLOGIQUE")
print("=" * 70)

print(summary)

print("\n")
print("=" * 70)
print("VALIDATION PREDATEUR-PROIES")
print("=" * 70)

print(df_inter)


## Test de correlation avec les Phi

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10,8)
)

for ax, sp in zip(
    axes.ravel(),
    species_names
):

    phi_name = f"Phi_{sp}"

    rho, p = spearmanr(
        arrays[phi_name],
        arrays[sp]
    )

    ax.scatter(
        arrays[phi_name],
        arrays[sp],
        s=2,
        alpha=0.2
    )

    ax.set_title(
        f"{species_labels[sp]}\n"
        f"ρ={rho:.2f}"
    )

    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()



### Calcul de la distance mediane entre chaque paire d'espèce
## On peut alors comparer avec les valeurs empiriques venant des GPS

def get_cell_centers(mesh):

    centers = vtk.vtkCellCenters()

    centers.SetInputData(mesh)

    centers.Update()

    pts = vtk_to_numpy(
        centers.GetOutput()
        .GetPoints()
        .GetData()
    )

    return (
        pts[:,0],
        pts[:,1]
    )


mesh = read_vtu(final_file)
mesh = point_to_cell(mesh)

areas = compute_areas(mesh)

X, Y = get_cell_centers(mesh)

import numpy as np

def sampled_median_distance(
    density_A,
    density_B,
    X,
    Y,
    areas,
    n_samples=5000,
    seed=123
):

    rng = np.random.default_rng(seed)

    # Probabilités pondérées par la masse
    pA = density_A * areas
    pA = pA / pA.sum()

    pB = density_B * areas
    pB = pB / pB.sum()

    coords = np.column_stack([X, Y])

    # Tirage de cellules selon les densités
    idx_A = rng.choice(
        len(coords),
        size=n_samples,
        replace=True,
        p=pA
    )

    idx_B = rng.choice(
        len(coords),
        size=n_samples,
        replace=True,
        p=pB
    )

    pts_A = coords[idx_A]
    pts_B = coords[idx_B]

    # Distance entre chaque paire tirée
    distances = np.sqrt(
        (pts_A[:,0] - pts_B[:,0])**2 +
        (pts_A[:,1] - pts_B[:,1])**2
    )

    return np.median(distances)

dist_CM = sampled_median_distance(
    arrays["C"],
    arrays["M"],
    X,
    Y,
    areas
)

dist_CP = sampled_median_distance(
    arrays["C"],
    arrays["P"],
    X,
    Y,
    areas
)

dist_CD = sampled_median_distance(
    arrays["C"],
    arrays["D"],
    X,
    Y,
    areas
)

dist_MP = sampled_median_distance(
    arrays["M"],
    arrays["P"],
    X,
    Y,
    areas
)

dist_MD = sampled_median_distance(
    arrays["M"],
    arrays["D"],
    X,
    Y,
    areas
)

dist_PD = sampled_median_distance(
    arrays["P"],
    arrays["D"],
    X,
    Y,
    areas
)

print(dist_CM)
print(dist_CP)
print(dist_CD)
print(dist_MP)
print(dist_MD)
print(dist_PD)


# print(X.min(), X.max())
# print(Y.min(), Y.max())

# print(np.column_stack([X, Y])[:10])

# print(len(np.unique(idx_A)))
# print(len(np.unique(idx_B)))

