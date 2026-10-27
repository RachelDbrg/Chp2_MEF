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
# LECTURE PARAMETRES .txt
# ============================================================

def read_parameters(txtfile):

    params = {}

    with open(txtfile, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            line = line.replace("=", " ")

            parts = line.split()

            if len(parts) >= 2:

                key = parts[0]

                try:
                    value = float(parts[1])
                    params[key] = value
                except:
                    pass

    return params

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




## Step 1: on parcourt les dossiers avec les outputs.vtu


root_dir = (
    "/users/rdubourg/Documents/Chp2_MEF/"
    "simulations/var_parms/i/Null_NMP_"
)

scenario_dirs = sorted(
    glob.glob(
        os.path.join(
            root_dir,
            "scenario_*"
        )
    )
)


## 2. Recupere les valeurs pour les mettre dans un df
results = []



## DEBUG: fait seulement sur les 3 premiers scenarios
for scenario_dir in scenario_dirs[:3]:

    ## Recupere les valeurs des parametres IFD et LOF
    param_file = glob.glob(
    os.path.join(
        scenario_dir,
        "*.txt"
    )
    )[0]

    params = read_parameters(
        param_file
    )

    print(params)


    c_ifd = (
        params.get("c1",0)
        + params.get("c2",0)
    )

    c_lof = (
        params.get("c3",0)
        + params.get("c4",0)
        + params.get("c5",0)
        + params.get("c6",0)
    )

    m_ifd = (
        params.get("m1",0)
        + params.get("m2",0)
    )

    m_lof = (
        params.get("m3",0)
        + params.get("m4",0)
        + params.get("m5",0)
        + params.get("m6",0)
)



    scenario_name = os.path.basename(
        scenario_dir
    )

    print(
        f"\nTraitement {scenario_name}"
    )

    nmp_files = sorted(
        glob.glob(
            os.path.join(
                scenario_dir,
                "snapshot_NMP_*.cont.vtu"
            )
        ),
        key=extract_time
    )

    nmpd_files = sorted(
        glob.glob(
            os.path.join(
                scenario_dir,
                "snapshot_NMPD_*.cont.vtu"
            )
        ),
        key=extract_time
    )

    for scen_type, files in [
        ("NMP", nmp_files),
        ("NMPD", nmpd_files)
    ]:

        for file in files:

            t = extract_time(file)

            BC = compute_bc_matrix(file)

            for pair, value in BC.items():

                results.append(
                    {
                        "scenario": scenario_name,
                        "simulation": scen_type,
                        "time": t,
                        "pair": pair,
                        "BC": value,

                        "c_ifd": c_ifd,
                        "c_lof": c_lof,

                        "m_ifd": m_ifd,
                        "m_lof": m_lof
                    }
                )


#df = pd.DataFrame(results)

# df.to_csv(
#     "Bhattacharyya_long.csv",
#     index=False
# )

# df_wide = (
#     df.pivot_table(
#         index=[
#             "scenario",
#             "simulation",
#             "time",
#             "c_ifd",
#             "c_lof",
#             "m_ifd",
#             "m_lof",
#             "p_ifd",
#             "p_lof"
#         ],
#         columns="pair",
#         values="BC"
#     )
#     .reset_index()
# )

# df.to_csv(
#     "Bhattacharyya_wide.csv",
#     index=False
# )


## ======================================================================
## Une fois que le .csv est formé, on peut faire les figures
## ======================================================================

markers = {
    "NMP": "o",
    "NMPD": "^"
}

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15,5),
    sharey=True
)

overlaps = {
    "C-M":"Blues",
    "C-P":"Reds",
    "C-D":"Greens"
}

for ax, (ov, cmap) in zip(
    axes,
    overlaps.items()
):

    for sim in ["NMP","NMPD"]:

        sub = df[
            df["simulation"] == sim
        ]

        sc = ax.scatter(
            sub["c_ratio"],
            sub[ov],
            c=sub["time"],
            cmap=cmap,
            marker=markers[sim],
            s=90,
            edgecolors="black"
        )

    ax.set_title(ov)
    ax.set_xlabel(
        r"$C_{IFD}/C_{LOF}$"
    )

axes[0].set_ylabel(
    "Bhattacharyya overlap"
)

cbar = fig.colorbar(
    sc,
    ax=axes,
    shrink=0.8
)

cbar.set_label("Time")

plt.tight_layout()
plt.show()