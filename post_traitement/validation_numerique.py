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
# STYLE GLOBAL DES FIGURES
# ============================================================

plt.rcParams.update({

    # Taille générale
    "figure.dpi": 150,
    "savefig.dpi": 300,

    # Police
    "font.size": 16,

    # Titres
    "axes.titlesize": 18,
    "axes.titleweight": "bold",

    # Labels axes
    "axes.labelsize": 18,

    # Ticks
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,

    # Légendes
    "legend.fontsize": 15,
    "legend.frameon": False,

    # Épaisseur des axes
    "axes.linewidth": 1.5,

    # Ticks
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,

    # Police PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42

})


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

## Modele decouple (sur 10 pas de temps seulement)
root_dir = "/users/rdubourg/Documents/Chp2_MEF/simulations/i/CMPD_/scenario_0000"
#root_dir = "/users/rdubourg/Documents/Chp2_MEF/simulations/validation_bio/CMPD_/scenario_0000"

files = sorted(
    glob.glob(
        os.path.join(
            root_dir,
            "*.cont.vtu"
        )
    ),
    key=extract_time
)



## Est ce qu'on regarde un modèle couplé?
MODEL_COUPLED = True


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

mass_results = []

for file in files:

    t = extract_time(file)

    data, areas = get_species_from_vtu(file)

    row = {
        "time": t
    }

    for sp in species_names:

        row[f"mass_{sp}"] = total_mass(
            data[sp],
            areas
        )

    mass_results.append(row)

df_mass = pd.DataFrame(mass_results)

mass_results = []



print(df_mass.head())
print(df_mass.columns)
print(df_mass.shape)



# ============================================================
# NORMALISATION
# ============================================================

## Pour tests de la vvalidation
for sp in species_names:

    m0 = df_mass[f"mass_{sp}"].iloc[0]

    df_mass[f"rel_mass_{sp}"] = (
        df_mass[f"mass_{sp}"] / m0
    )


# =====================
# Verification de la variation dans la masse 
# =====================

for sp in species_names:

    mass = df_mass[f"mass_{sp}"].values

    rel_dM = np.diff(mass) / mass[0]

    print(
        sp,
        "max rel ΔM =",
        np.abs(rel_dM).max()
    )

# ============================================================
# ERREUR MAXIMALE
# ============================================================

print("\nErreur maximale de conservation")

for sp in species_names:

    err = np.max(

        np.abs(

            df_mass[
                f"rel_mass_{sp}"
            ]

            - 1.0

        )

    )

    print(
        f"{sp} : {err:.3e}"
    )


for sp in species_names:

    print(
        sp,
        df_mass[f"rel_mass_{sp}"].min(),
        df_mass[f"rel_mass_{sp}"].max()
    )





# ============================================================
# FIGURE - validation de la masse dans le temps
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,5),
    sharey=True
)

for ax, sp in zip(
    axes.ravel(),
    species_names
):
    
    tmp = (
        df_mass
        .groupby("time")
        [f"rel_mass_{sp}"]
        .mean()
        .reset_index()
    )

    ax.plot(
        tmp["time"],
        tmp[f"rel_mass_{sp}"],
        color="black",
        linewidth=3
    )

    ax.axhline(
        1.0,
        color="red",
        linestyle="--"
    )

    ax.tick_params(
    axis="both",
    which="major",
    labelsize=15,
    width=1.5,
    length=6
)

    ax.set_title(
    species_labels[sp])

    ax.set_xlabel("Time")

axes[0,0].set_ylabel(
    r"$mass(t)/mass(0)$"
)

axes[1,0].set_ylabel(
    r"$mass(t)/mass(0)$"
)

plt.tight_layout()
plt.show()




# ============================================================
# VALIDATION NUMERIQUE
# Diffusion analytique + transport du centre de masse
# ============================================================

def analytical_gaussian(
    x,
    y,
    t,
    amplitude,
    etalement,
    D,
    x0=0.5,
    y0=0.5
):

    sigma0_sq = 1.0 / (2.0 * etalement)

    sigma_t_sq = (
        sigma0_sq
        +
        2.0 * D * t
    )

    r2 = (
        (x - x0)**2
        +
        (y - y0)**2
    )

    return (

        amplitude

        * sigma0_sq
        / sigma_t_sq

        * np.exp(
            -r2
            /
            (2.0 * sigma_t_sq)
        )

    )


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


def center_of_mass(
    density,
    x,
    y,
    areas
):

    m = np.sum(
        density * areas
    )

    xcm = np.sum(
        x * density * areas
    ) / m

    ycm = np.sum(
        y * density * areas
    ) / m

    return (
        xcm,
        ycm
    )


# ============================================================
# PARAMETRES DE REFERENCE
# ============================================================

etalementCloche = 300

species_params = {

    "C": {
        "amplitude": 1.0,
        "D": 1.0
    },

    "M": {
        "amplitude": 1.0,
        "D": 142.35 / 1073.1
    },

    "P": {
        "amplitude": 1.0,
        "D": 1051.25 / 1073.1
    },

    "D": {
        "amplitude": 1.0,
        "D": 142.35 / 1073.1
    }

}

delta_t = 0.01

diffusion_results = []

com_results = []


# ============================================================
# BOUCLE SUR LES SORTIES VTU
# ============================================================

for file in files:

    step = extract_time(file)

    t = step * delta_t

    mesh = read_vtu(file)

    mesh = point_to_cell(mesh)

    areas = compute_areas(mesh)

    x,y = get_cell_centers(mesh)

    cell_data = mesh.GetCellData()

    row_com = {
        "time": t
    }

    for sp in species_names:

        U_num = vtk_to_numpy(
            cell_data.GetArray(sp)
        )

        # ----------------------------------------------------
        # Solution analytique
        # ----------------------------------------------------

        U_exact = analytical_gaussian(

            x=x,
            y=y,

            t=t,

            amplitude=
                species_params[sp]["amplitude"],

            etalement=
                etalementCloche,

            D=
                species_params[sp]["D"]

        )

        err = U_num - U_exact

        L1 = np.mean(
            np.abs(err)
        )

        L2 = np.sqrt(
            np.mean(err**2)
        )

        Linf = np.max(
            np.abs(err)
        )

        diffusion_results.append({

            "species": sp,

            "time": t,

            "L1": L1,

            "L2": L2,

            "Linf": Linf

        })

        # ----------------------------------------------------
        # Centre de masse
        # ----------------------------------------------------

        xcm, ycm = center_of_mass(

            density=U_num,

            x=x,
            y=y,

            areas=areas

        )

        row_com[f"xcm_{sp}"] = xcm
        row_com[f"ycm_{sp}"] = ycm

    com_results.append(
        row_com
    )


# ============================================================
# DATAFRAMES
# ============================================================

df_diffusion = pd.DataFrame(
    diffusion_results
)

df_com = pd.DataFrame(
    com_results
)


# ============================================================
# FIGURE 1
# ERREUR ANALYTIQUE - diffusion
# ============================================================

fig, ax = plt.subplots(
    figsize=(9,6)
)

for sp in species_names:

    sub = df_diffusion[
        df_diffusion["species"] == sp
    ]

    ax.semilogy(
        sub["time"],
        sub["L2"],
        linewidth=3,
        label=species_labels[sp]
    )
    ax.grid(
    alpha=0.3
    )

    ax.tick_params(
    axis="both",
    which="major",
    labelsize=15,
    width=1.5,
    length=6
)

ax.set_xlabel("Time")

ax.set_ylabel(
    r"$L_2$ error"
)

ax.legend(
    loc="best",
    fontsize=15
)

plt.tight_layout()

plt.show()


# ============================================================
# FIGURE 2
# DEPLACEMENT DU CENTRE DE MASSE
# ============================================================
ymin = df_com["ycm_D"].min()
ymax = df_com["ycm_D"].max()

padding = 0.05 * (ymax - ymin)

ymin -= padding
ymax += padding


fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,9),
    sharex=True,
    sharey=True
)

for ax, sp in zip(
    axes.ravel(),
    species_names
):

    y0 = df_com[f"ycm_{sp}"].iloc[0]

    ax.plot(
        df_com["time"],
        df_com[f"ycm_{sp}"] - y0,
        color="black",
        linewidth=3
    )


    ax.axhline(
        0,
        ls="--",
        color="grey"
    )

    ax.set_title(
        species_labels[sp]
    )


    ax.tick_params(
    axis="both",
    which="major",
    labelsize=15,
    width=1.5,
    length=6
    )

for ax in axes[:,0]:
    ax.set_ylabel(
        r"$y_{CM}(t)-y_{CM}(0)$"
    )

for ax in axes[1,:]:
    ax.set_xlabel("Time")

plt.tight_layout()
plt.show()



# ============================================================
# RESUME NUMERIQUE
# ============================================================

print("\n")
print("="*70)
print("NUMERICAL VALIDATION")
print("="*70)

print("\n")
print("Diffusion errors")
print("-"*70)

for sp in species_names:

    sub = df_diffusion[
        df_diffusion["species"] == sp
    ]

    print(

        f"{species_labels[sp]} : "
        f"L2(final) = "
        f"{sub['L2'].iloc[-1]:.3e}"

    )


print("\n")
print("Centre of mass displacement")
print("-"*70)

for sp in species_names:

    dx = (

        df_com[f"xcm_{sp}"].iloc[-1]

        -

        df_com[f"xcm_{sp}"].iloc[0]

    )

    dy = (

        df_com[f"ycm_{sp}"].iloc[-1]

        -

        df_com[f"ycm_{sp}"].iloc[0]

    )

    print(

        f"{species_labels[sp]} : "
        f"Δx = {dx:.4e}   "
        f"Δy = {dy:.4e}"

    )