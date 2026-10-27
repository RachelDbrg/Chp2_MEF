import os
import re
import glob
import itertools

import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy


# ============================================================
# PARAMETRES
# ============================================================

species_names = [
    "C",
    "M",
    "P",
    "D"
]


# ============================================================
# UTILITAIRES
# ============================================================

def extract_time(filename):

    m = re.search(r"_(\d+)\.cont\.vtu$", filename)

    return int(m.group(1))


def bhattacharyya(pi, pj, areas):

    Mi = np.sum(pi * areas)
    Mj = np.sum(pj * areas)

    if Mi == 0 or Mj == 0:
        return np.nan

    pi = pi / Mi
    pj = pj / Mj

    return np.sum(np.sqrt(pi * pj) * areas)


# ============================================================
# LECTURE VTU
# ============================================================

def read_vtu(filename):

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    return reader.GetOutput()


# ============================================================
# POINT -> CELL
# ============================================================

def point_to_cell(mesh):

    filt = vtk.vtkPointDataToCellData()
    filt.SetInputData(mesh)
    filt.Update()

    return filt.GetOutput()


# ============================================================
# AIRE DES CELLULES
# ============================================================

def compute_areas(mesh):

    quality = vtk.vtkCellSizeFilter()
    quality.SetInputData(mesh)
    quality.ComputeAreaOn()
    quality.Update()

    out = quality.GetOutput()

    areas = vtk_to_numpy(
        out.GetCellData().GetArray("Area")
    )

    return areas


# ============================================================
# OVERLAP
# ============================================================

def compute_bc_matrix(filename):

    mesh = read_vtu(filename)

    mesh = point_to_cell(mesh)

    areas = compute_areas(mesh)

    cell_data = mesh.GetCellData()

    species = []

    for name in species_names:

        arr = vtk_to_numpy(
            cell_data.GetArray(name)
        )

        species.append(arr)

    overlaps = {}

    for i, j in itertools.combinations(range(4), 2):

        bc = bhattacharyya(
            species[i],
            species[j],
            areas
        )

        overlaps[
            f"{species_names[i]}-{species_names[j]}"
        ] = bc

    return overlaps


# ============================================================
# FICHIERS
# ============================================================

results = []

data_dir = "/users/rdubourg/Documents/Chp2_MEF/simulations/var_parms/i/Null_NMP_/scenario_0000"

nmp_files = sorted(
    glob.glob(
        os.path.join(
            data_dir,
            "snapshot_NMP_*.cont.vtu"
        )
    ),
    key=extract_time
)

nmpd_files = sorted(
    glob.glob(
        os.path.join(
            data_dir,
            "snapshot_NMPD_*.cont.vtu"
        )
    ),
    key=extract_time
)

print("NMP :", len(nmp_files))
print("NMPD :", len(nmpd_files))



all_files = [
    ("NMP", nmp_files),
    ("NMPD", nmpd_files)
]

for scenario, files in all_files:

    print(f"{scenario}: {len(files)} fichiers trouvés")

    for n, file in enumerate(files):

        print(
            f"{scenario} : {n+1}/{len(files)}"
        )

        t = extract_time(file)

        BC = compute_bc_matrix(file)

        for pair, value in BC.items():

            results.append(
                {
                    "scenario": scenario,
                    "time": t,
                    "pair": pair,
                    "BC": value
                }
            )

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    "Bhattacharyya_timeseries.csv",
    index=False
)

print(df.head())


# ============================================================
# FIGURES
# ============================================================

pairs = sorted(df["pair"].unique())

for pair in pairs:

    plt.figure(figsize=(8,5))

    tmp = df[df["pair"] == pair]

    for scenario in ["NMP","NMPD"]:

        sub = tmp[tmp["scenario"] == scenario]

        plt.plot(
            sub["time"],
            sub["BC"],
            lw=2,
            label=scenario
        )

    plt.xlabel("Temps")
    plt.ylabel("Bhattacharyya overlap")
    plt.title(f"Overlap {pair}")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"Compare_{pair}.png",
        dpi=300
    )

    plt.close()


plt.figure(figsize=(10,6))

for pair in pairs:

    nmp = (
        df[
            (df["scenario"]=="NMP")
            & (df["pair"]==pair)
        ]
        .sort_values("time")
    )

    nmpd = (
        df[
            (df["scenario"]=="NMPD")
            & (df["pair"]==pair)
        ]
        .sort_values("time")
    )

    delta = nmpd["BC"].values - nmp["BC"].values

    plt.plot(
        nmp["time"],
        delta,
        label=pair
    )

plt.axhline(0,color='k',linestyle='--')

plt.xlabel("Temps")
plt.ylabel("ΔBC = BC(NMPD)-BC(NMP)")
plt.legend()
plt.tight_layout()

plt.savefig(
    "Delta_BC_AllPairs.png",
    dpi=300
)

print("Terminé.")