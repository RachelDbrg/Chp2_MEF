import os
import re
import glob
import itertools

import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy

from matplotlib.lines import Line2D



# ============================================================
# PARAMETRES
# ============================================================


## Noms des espèces du système utilisés pour les overlaps entre espèces animales
species_names = [
    "C",
    "M",
    "P",
    "D"
]

## Noms des ressources, utilisés pour calculer overlap animal-ressource
resource_map = {

    "C": [
        "M1_C","M2_C","M3_C",
        "M4_C","M5_C","M6_C",
        "Phi_C"
    ],

    "M": [
        "M1_M","M2_M","M3_M",
        "M4_M","M5_M","M6_M",
        "Phi_M"
    ],

    "P": [
        "M4_P","M5_P","M6_P",
        "Phi_P"
    ],

    "D": [
        "M1_D","M2_D","M3_D",
        "M4_D","M5_D","M6_D",
        "Phi_D"
    ]
}

# ============================================================
# UTILITAIRES
# ============================================================

def extract_time(filename):

    m = re.search(
        r"_(\d+)\.cont\.vtu$",
        filename
    )

    return int(m.group(1))


def bhattacharyya(pi, pj, areas):

    Mi = np.sum(pi * areas)
    Mj = np.sum(pj * areas)

    if Mi == 0 or Mj == 0:
        return np.nan

    pi = pi / Mi
    pj = pj / Mj

    return np.sum(np.sqrt(pi * pj) * areas)


def occupied_fraction(
    density,
    areas,
    fraction=0.1
):

    threshold = (
        fraction * density.max()
    )

    occupied_area = np.sum(
        areas[density > threshold]
    )

    return (
        occupied_area
        /
        np.sum(areas)
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


def relative_l1_difference(
    f_cmp,
    f_cmpd,
    areas
):

    num = np.sum(
        np.abs(
            f_cmpd - f_cmp
        ) * areas
    )

    den = np.sum(
        np.abs(
            f_cmp
        ) * areas
    )

    if den < 1e-15:
        return np.nan

    return num / den

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
# OVERLAP - metriques
# ============================================================

## Overlap absolu   
def absolute_overlap(pi, pj, areas):

    return np.sum(
        np.sqrt(pi * pj) * areas
    )

## Overlap pondéré
def weighted_overlap(pi, pj, areas):

    return np.sum(
        pi * pj * areas
    )


## Overlap espèce animale - ressource (selon le potentiel Phi)
# ============================================================
# ALIGNEMENT ESPECE - POTENTIEL
# ============================================================

def resource_alignment(
    density,
    potential,
    areas
):

    numerator = np.sum(
        density * potential * areas
    )

    norm_density = np.sqrt(
        np.sum(
            density**2 * areas
        )
    )

    norm_potential = np.sqrt(
        np.sum(
            potential**2 * areas
        )
    )

    if norm_density < 1e-15:
        return np.nan

    if norm_potential < 1e-15:
        return np.nan

    return (
        numerator
        /
        (norm_density * norm_potential)
    )



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

    results = {}

    for i, j in itertools.combinations(range(4), 2):

        pair = f"{species_names[i]}-{species_names[j]}"

        bc = bhattacharyya(
            species[i],
            species[j],
            areas
        )

        abs_overlap = absolute_overlap(
            species[i],
            species[j],
            areas
        )

        weighted = weighted_overlap(
            species[i],
            species[j],
            areas
        )

        results[pair] = {
            "BC": bc,
            "ABS": abs_overlap,
            "WGT": weighted
        }

    return results



def compute_resource_alignment(filename):

    mesh = read_vtu(filename)

    mesh = point_to_cell(mesh)

    areas = compute_areas(mesh)

    cell_data = mesh.GetCellData()

    results = []

    

    ## Ici on calcule l'overlap avec les ressources 
    for sp, resources in resource_map.items():

        sp_array = vtk_to_numpy(
            cell_data.GetArray(sp)
        )

        for resource in resources:

            arr = cell_data.GetArray(
                resource
            )

            if arr is None:
                continue

            res_array = vtk_to_numpy(arr)


            align = resource_alignment(
                sp_array,
                res_array,
                areas
            )

            results.append(
                {
                    "pair":
                    f"{sp}-{resource}",

                    "ALIGN":
                    align
                }
            )

    ## Ici on calcule l'overlap avec le prédateur
    predator_pairs = {
    "C": "P",
    "M": "P",
    "D": "P"
    }

    ## Ici, on considere differentes facons de modéliser le risque de predation
    risk_fields = {

    "C": {
        "riskP": ["P"],
        "riskPM": ["P","M"],
        "riskPMD": ["P","M","D"]
    },

    "M": {
        "riskP": ["P"],
        "riskPC": ["P","C"],
        "riskPCD": ["P","C","D"]
    },

    "D": {
        "riskP": ["P"],
        "riskPM": ["P","M"],
        "riskPMC": ["P","M","C"]
    }
}

    for prey, predator in predator_pairs.items():

        prey_arr = vtk_to_numpy(
            cell_data.GetArray(prey)
        )

        pred_arr = vtk_to_numpy(
            cell_data.GetArray(predator)
        )

        align_pred = resource_alignment(
            prey_arr,
            pred_arr,
            areas
        )

        results.append(
            {
                "pair": f"{prey}-{predator}_risk",
                "ALIGN": align_pred
            }
        )

    for focal, risks in risk_fields.items():

        focal_arr = vtk_to_numpy(
            cell_data.GetArray(focal)
        )

        for risk_name, components in risks.items():

            risk_arr = np.zeros_like(
                focal_arr
            )

            for comp in components:

                risk_arr += vtk_to_numpy(
                    cell_data.GetArray(comp)
                )

            align = resource_alignment(
                focal_arr,
                risk_arr,
                areas
            )

            results.append(
                {
                    "pair":
                    f"{focal}-{risk_name}",
                    "ALIGN":
                    align
                }
            )



    return results



## Step 1: on parcourt les dossiers avec les outputs.vtu

root = "/users/rdubourg/Documents/Chp2_MEF/simulations"

simulation_sets = []

for experiment in sorted(os.listdir(root)):

    experiment_dir = os.path.join(
        root,
        experiment
    )

    if not os.path.isdir(experiment_dir):
        continue

    for sim in ["CMP", "CMPD"]:

        sim_dir = os.path.join(
            experiment_dir,
            f"{sim}_"
        )

        if os.path.isdir(sim_dir):

            if experiment == "i":
                experiment_type = "baseline"

            # elif experiment == "test_figures":
            #     experiment_type = "alternative"


            ## Cherche dans le sous-dossier ii) pour lequel on a fait varier les parms 
            elif experiment == "ii":
                experiment_type = "alternative"

            else:
                continue

            simulation_sets.append(
                {
                    "parameter_set": experiment,
                    "simulation": sim,
                    "path": sim_dir,
                    "experiment_type": experiment_type
                }
            )

## Verification
print(pd.DataFrame(simulation_sets))

## Figure pour l'indice BC
colors = {
    "CMP": "tab:blue",
    "CMPD": "tab:red"
}
markers = {
    "CMP": "o",
    "CMPD": "^"
}


## 2. Recupere les valeurs pour les mettre dans un df
results = []
spread_results = []


for sim_info in simulation_sets:

    parameter_set = sim_info["parameter_set"]
    scen_type = sim_info["simulation"]
    experiment_type = sim_info["experiment_type"]

    scenario_list = sorted(
        glob.glob(
            os.path.join(
                sim_info["path"],
                "scenario_*"
            )
        )
    )

    print(
        parameter_set,
        scen_type,
        experiment_type,
        len(scenario_list)
    )


    # =========== MODE TEST
    if experiment_type == "alternative":
        scenario_list = scenario_list[:2]
    # =========== MODE TEST

 
    for scenario_dir in scenario_list:
 
        scenario_name = os.path.basename(
            scenario_dir
        )

        print(
        f"\nTraitement {scen_type} "
        f"{scenario_name}"
        )



        param_files = glob.glob(
            os.path.join(
                scenario_dir,
                "*.txt"
            )
        )

        if len(param_files) == 0:
            print("No parameter file")
            continue


        ## C'est a cette étape qu'on recupere les valeurs des parms pour ensuite calculer clof, icfd etc
        params = read_parameters(
            param_files[0]
        )

        files = sorted(
            glob.glob(
                os.path.join(
                    scenario_dir,
                    "*.cont.vtu"
                )
            ),
            key=extract_time
        )


        ## On construit les metriques de LOF et d'IFD

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

        p_ifd = (
                params.get("p1",0)
                + params.get("p2",0)
            )
        
        p_lof = (
            params.get("p3",0)
            + params.get("p4",0)
            + params.get("p5",0)
            + params.get("p6",0)
        )



        for file in files:

                t = extract_time(file)

                mesh = read_vtu(file)

                mesh = point_to_cell(mesh)

                areas = compute_areas(mesh)

                cell_data = mesh.GetCellData()

                for sp in ["C","M","P","D"]:

                    density = vtk_to_numpy(
                        cell_data.GetArray(sp)
                    )

                    spread = occupied_fraction(
                        density,
                        areas
                    )

                    spread_results.append(
                        {
                            "parameter_set": parameter_set,
                            "experiment_type": experiment_type,
                            "scenario": scenario_name,
                            "simulation": scen_type,
                            "time": t,
                            "species": sp,
                            "spread": spread
                        }
                    )





                BC = compute_bc_matrix(file)

                
                ## Verification de si on obtient ou pas des trucs différents
                if t == max(extract_time(f) for f in files):
                    print(
                        scen_type,
                        experiment_type,
                        scenario_name,
                        BC["C-P"]["BC"]
                    )

                resource_alignment_results = (
                compute_resource_alignment(
                    file
                )
            )

                for pair, values in BC.items():

                    results.append(
                        {
                            "parameter_set": parameter_set,
                            "experiment_type": experiment_type,
                            "scenario": scenario_name,
                            "simulation": scen_type,
                            "time": t,
                            "pair": pair,
                            
                            "BC": values["BC"],
                            "ABS": values["ABS"],
                            "WGT": values["WGT"],

                            "c_ifd": c_ifd,
                            "c_lof": c_lof,

                            "m_ifd": m_ifd,
                            "m_lof": m_lof,

                            "p_ifd": p_ifd,
                            "p_lof": p_lof
                        }
                    )


                for item in resource_alignment_results:

                    results.append(
                        {
                            "parameter_set": parameter_set,
                            "experiment_type": experiment_type,
                            "scenario": scenario_name,
                            "simulation": scen_type,
                            "time": t,

                            "pair": item["pair"],

                            "BC": np.nan,
                            "ABS": np.nan,
                            "WGT": np.nan,

                            "ALIGN": item["ALIGN"],

                            "c_ifd": c_ifd,
                            "c_lof": c_lof,

                            "m_ifd": m_ifd,
                            "m_lof": m_lof,

                            "p_ifd": p_ifd,
                            "p_lof": p_lof
                        }
                    )

                





df = pd.DataFrame(results)


print("\nSimulations présentes :")
print(df["simulation"].value_counts())

print("\nValeurs uniques :")
print(df["simulation"].unique())

print("\nQuelques lignes CMPD :")
print(
    df[df["simulation"] == "CMPD"]
    .head(10)
)


df.to_csv(
    "Bhattacharyya_long.csv",
    index=False
)

print(df.columns)
print(df.head())
print(df.shape)


index_cols = [
    "parameter_set",
    "experiment_type",
    "scenario",
    "simulation",
    "time",
    "c_ifd",
    "c_lof",
    "m_ifd",
    "m_lof",
    "p_ifd",
    "p_lof"
]


df_bc = (
    df.pivot_table(
        index=index_cols,
        columns="pair",
        values="BC"
    )
    .reset_index()
)

df_bc = df_bc.rename(
    columns={
        c: f"{c}_BC"
        for c in df_bc.columns
        if c not in index_cols
    }
)


df_abs = (
    df.pivot_table(
        index=index_cols,
        columns="pair",
        values="ABS"
    )
    .reset_index()
)

df_abs = df_abs.rename(
    columns={
        c: f"{c}_ABS"
        for c in df_abs.columns
        if c not in index_cols
    }
)

df_wgt = (
    df.pivot_table(
        index=index_cols,
        columns="pair",
        values="WGT"
    )
    .reset_index()
)

df_wgt = df_wgt.rename(
    columns={
        c: f"{c}_WGT"
        for c in df_wgt.columns
        if c not in index_cols
    }
)


## Partie sur l'alignement avec les ressources
df_align = (
    df.pivot_table(
        index=index_cols,
        columns="pair",
        values="ALIGN"
    )
    .reset_index()
)

df_align = df_align.rename(
    columns={
        c: f"{c}_ALIGN"
        for c in df_align.columns
        if c not in index_cols
    }
)

df_wide = (
    df_bc
    .merge(df_abs, on=index_cols, how="outer")
    .merge(df_wgt, on=index_cols, how="outer")
    .merge(df_align, on=index_cols, how="outer")
)



## Verification
print(df_wide["simulation"].value_counts())


df_wide["c_ratio"] = np.where(
    df_wide["c_lof"] > 0,
    df_wide["c_ifd"] / df_wide["c_lof"],
    np.nan
)

print(df_wide["c_ratio"].describe())

print(
    df_wide["c_ratio"]
    .drop_duplicates()
    .sort_values()
    .values
)


## Calcul de plusieurs ratio selon comment est considere le risque de predation
### Caribou
df_wide["C_ratio_P"] = (
    df_wide["C-Phi_C_ALIGN"]
    /
    df_wide["C-riskP_ALIGN"]
)

df_wide["C_ratio_PM"] = (
    df_wide["C-Phi_C_ALIGN"]
    /
    df_wide["C-riskPM_ALIGN"]
)

df_wide["C_ratio_PMD"] = (
    df_wide["C-Phi_C_ALIGN"]
    /
    df_wide["C-riskPMD_ALIGN"]
)


### Orignal
df_wide["M_ratio_P"] = (
    df_wide["M-Phi_M_ALIGN"]
    /
    df_wide["M-riskP_ALIGN"]
)

df_wide["M_ratio_PC"] = (
    df_wide["M-Phi_M_ALIGN"]
    /
    df_wide["M-riskPC_ALIGN"]
)

df_wide["M_ratio_PCD"] = (
    df_wide["M-Phi_M_ALIGN"]
    /
    df_wide["M-riskPCD_ALIGN"]
)

### Cerf
df_wide["D_ratio_P"] = (
    df_wide["D-Phi_D_ALIGN"]
    /
    df_wide["D-riskP_ALIGN"]
)

df_wide["D_ratio_PM"] = (
    df_wide["D-Phi_D_ALIGN"]
    /
    df_wide["D-riskPM_ALIGN"]
)

df_wide["D_ratio_PMC"] = (
    df_wide["D-Phi_D_ALIGN"]
    /
    df_wide["D-riskPMC_ALIGN"]
)

df_wide["log_ratio"] = np.log10(
    df_wide["c_ifd"] /
    df_wide["c_lof"]
)


df_wide.to_csv(
    "Bhattacharyya_wide.csv",
    index=False
)

baseline = df_wide[
    df_wide["experiment_type"] == "baseline"
]

alternative = df_wide[
    df_wide["experiment_type"] == "alternative"
]

baseline_cmp = baseline[
    baseline["simulation"] == "CMP"
]

alternative_cmp = alternative[
    alternative["simulation"] == "CMP"
]


# ============================================================
# Creation d'identifiants SC0, SC1, ..., SC9
# ============================================================


scenario_names = sorted(
    df_wide["scenario"].unique()
)

scenario_order = (
    df_wide
    .groupby("scenario")["log_ratio"]
    .mean()
    .sort_values()
    .index
)

scenario_map = {
    scen: f"SC{i}"
    for i, scen in enumerate(scenario_order)
}

print(scenario_map) 



df_wide["SC"] = (
    df_wide["scenario"]
    .map(scenario_map)
)

df_wide["SC_label"] = (
    df_wide["simulation"]
    + "_"
    + df_wide["SC"]
)


print(
    df_wide[
        ["parameter_set","simulation"]
    ]
    .drop_duplicates()
    .sort_values(
        ["parameter_set","simulation"]
    )
)



## ==============================================================================
## FIGURE 1: pairwise overlap a 3 moments de la simulation (debut, milieu et fin)
## Comparaison entre composition et experiment_type (baseline vs. alt)
## ==============================================================================

## Utiliser les experiment_type pour distringuer entre baseline et alt
## Prendre 3 time steps (diviser la duree de sim/3)

## Ici, on mesure a quel point la forme spatiale relative des distributions
## entre les paires d'espèce varie dans le temps 


# ============================================================
# 5 time steps - car le plus gros des différencse
# a l,air de se produire entre tinit et tfinal
# ============================================================

times = np.sort(baseline_cmp["time"].unique())

selected_times = [
    times[0],
    times[len(times)//4],
    times[len(times)//2],
    times[3*len(times)//4],
    times[-1]
]

pairs = [
    "C-M_BC",
    "C-P_BC",
    "M-P_BC"
]


# ============================================================
# Couleurs
# ============================================================

scenario_colormaps = {
    "CMP": plt.cm.Blues,
    "CMPD": plt.cm.Reds
}

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(15, 5),
    sharey=False
)

for ax, pair in zip(axes, pairs):

    for simulation in ["CMP", "CMPD"]:

        cmap = scenario_colormaps[simulation]

        # clair -> foncé avec le temps
        time_colors = cmap(
            np.linspace(0.35, 0.9, len(selected_times))
        )

        for color, t in zip(time_colors, selected_times):

            tmp = df_wide[
                (df_wide["time"] == t)
                & (df_wide["simulation"] == simulation)
            ]

            for experiment_type, sub in tmp.groupby(
                "experiment_type"
            ):

                marker = (
                    "o"
                    if experiment_type == "baseline"
                    else "^"
                )

                ax.scatter(
                    sub["log_ratio"],
                    sub[pair],
                    marker=marker,
                    c=[color],
                    edgecolors="black",
                    linewidth=0.7,
                    s=120,
                    alpha=0.9
                )

                if t == selected_times[-1]:

                    for _, row in sub.iterrows():

                        ax.annotate(
                            row["SC_label"],
                            (
                                row["log_ratio"],
                                row[pair]
                            ),
                            fontsize=7,
                            xytext=(3,3),
                            textcoords="offset points"
                        )


    ax.set_title(
        pair.replace("_BC", "")
    )

    ax.set_xlabel(
    r"$\log_{10}(C_{IFD}/C_{LOF})$"
)

    ax.grid(
        alpha=0.2,
        linestyle=":"
    )

axes[0].set_ylabel(
    "Bhattacharyya coefficient"
)

# ============================================================
# Légende : type d'expérience
# ============================================================

experiment_legend = [

    Line2D(
        [0], [0],
        marker='o',
        color='black',
        linestyle='',
        markersize=10,
        label='Baseline'
    ),

    Line2D(
        [0], [0],
        marker='^',
        color='black',
        linestyle='',
        markersize=10,
        label='Alternative'
    )
]

# ============================================================
# Légende : scénario
# ============================================================

scenario_legend = [

    Line2D(
        [0], [0],
        marker='o',
        color=plt.cm.Blues(0.75),
        linestyle='',
        markersize=10,
        label='CMP'
    ),

    Line2D(
        [0], [0],
        marker='o',
        color=plt.cm.Reds(0.75),
        linestyle='',
        markersize=10,
        label='CMPD'
    )
]

# ============================================================
# Légende : temps
# ============================================================

time_legend =  [

    Line2D(
        [0], [0],
        marker='o',
        color='grey',
        alpha=a,
        linestyle='',
        markersize=10,
        label=f"t = {t}"
    )

    for a, t in zip(
        [0.4, 0.7, 1.0],
        selected_times
    )
]

fig.legend(
    handles=time_legend,
    loc="lower center",
    ncol=len(selected_times)
)

axes[-1].legend(
    handles=(
        experiment_legend
        + scenario_legend
        + time_legend
    ),
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()
plt.show()


## ==============================================================================
## FIGURE 2: Comparaison dans tout le temps de la simulation dans les pairwise
## overlaps.
## ==============================================================================

## Indice BC
pairs = [
    p for p in [
        "C-M_BC",
        "C-P_BC",
        "C-D_BC",
        "M-P_BC",
        "M-D_BC"
    ]
    if p in df_wide.columns
]

fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(25,5),
    sharey=True
)

for ax, pair in zip(axes, pairs):

    for (sim, sc), sub in (
    df_wide.groupby(["simulation", "SC"])
):

        sub = sub.sort_values("time")

        # facecolor = (
        #     "none"
        #     if exp_type == "baseline"
        #     else colors[sim]
        # )

        ## On trace une ligne
        ax.plot(
            sub["time"],
            sub[pair],
            color=colors[sim],
            #linestyle="--" if exp_type == "baseline" else "-",
            linewidth=2,
            alpha=0.8
        )

        ax.text(
            sub["time"].iloc[-1],
            sub[pair].iloc[-1],
            sub["SC_label"].iloc[0],
            fontsize=7
        )

        ax.scatter(
            sub["time"],
            sub[pair],
            color=colors[sim],
            s=10
        )

    ax.set_title(
        pair.replace("_BC", "")
    )

    ax.set_xlabel("Temps")

axes[0].set_ylabel(
    "Bhattacharyya overlap"
)

from matplotlib.lines import Line2D

legend_items = [

    Line2D(
        [0],[0],
        marker='o',
        color='tab:blue',
        linestyle='',
        label='CMP'
    ),

    Line2D(
        [0],[0],
        marker='^',
        color='tab:red',
        linestyle='',
        label='CMPD'
    ),

    Line2D(
        [0],[0],
        marker='o',
        markerfacecolor='none',
        markeredgecolor='black',
        linestyle='',
        label='Baseline'
    ),

    Line2D(
        [0],[0],
        marker='o',
        markerfacecolor='black',
        markeredgecolor='black',
        linestyle='',
        label='Alternative'
    )
]

axes[-1].legend(
    handles=legend_items,
    loc="best"
)

plt.tight_layout()
plt.show()


## ======================================================================
## FIGURE 3: Delta BC entre alternatives et baseline
## ======================================================================

pairs = [
    p for p in [
        "C-M_BC",
        "C-P_BC",
        "C-D_BC",
        "M-P_BC",
        "M-D_BC"
    ]
    if p in df_wide.columns
]

baseline = (
    df_wide[
        df_wide["experiment_type"] == "baseline"
    ]
    .copy()
)

alternative = (
    df_wide[
        df_wide["experiment_type"] != "baseline"
    ]
    .copy()
)

merge_cols = [
    "scenario",
    "simulation",
    "time"
]

df_delta = alternative.merge(
    baseline,
    on=merge_cols,
    suffixes=("_alt", "_base"),
    validate="one_to_one"
)

## Verification de si ce truc n'est pas vide
print(df_delta.shape)
print(df_delta["simulation"].value_counts())
print(df_delta["time"].unique())

for pair in pairs:

    df_delta[f"delta_{pair}"] = (
        df_delta[f"{pair}_alt"]
        - df_delta[f"{pair}_base"]
    )

print(df_delta.shape)

print(
    df_delta[
        [
            "scenario",
            "simulation",
            "time"
        ]
    ].head()
)


colors = {
    "CMP": "tab:blue",
    "CMPD": "tab:red"
}

fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(25, 5),
    sharey=True
)

for ax, pair in zip(axes, pairs):

    delta_col = f"delta_{pair}"

    for sim in ["CMP", "CMPD"]:

        summary = (
            df_delta[
                df_delta["simulation"] == sim
            ]
            .groupby("time")[delta_col]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.plot(
            summary["time"],
            summary["mean"],
            color=colors[sim],
            linewidth=2,
            label=sim
        )

        ax.fill_between(
            summary["time"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=colors[sim],
            alpha=0.2
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        pair.replace("_BC", "")
    )

    ax.set_xlabel("Time")

axes[0].set_ylabel(
    r"$\Delta BC = BC_{alt} - BC_{base}$"
)

axes[-1].legend()

plt.tight_layout()
plt.show()



## ======================================================================
## QUAND EST CE QUE JE ME SERS DE CA?
## ======================================================================

## metrique recapitulative, coefficient de variation de BC
pairs = [
    "C-M_BC",
    "C-P_BC",
    "C-D_BC",
    "M-P_BC",
    "M-D_BC"
]

pairs = [p for p in pairs if p in df_wide.columns]

cv_results = []

for pair in pairs:

    for (pset, sim), sub in (
        df_wide.groupby(
            ["parameter_set","simulation"]
        )
    ):

        mean_bc = sub[pair].mean()
        std_bc = sub[pair].std()

        cv_bc = std_bc / mean_bc

        cv_results.append(
            {
                "parameter_set": pset,
                "simulation": sim,
                "pair": pair.replace("_BC",""),
                "mean_BC": mean_bc,
                "std_BC": std_bc,
                "CV_BC": cv_bc
            }
        )

df_cv = pd.DataFrame(cv_results)

print(df_cv)



## ======================================================================
## FIGURE 4: Delta d'overlap entre CMPD et CMP qui ont les memes valeurs
## parms (donc 1 courbe par couple de parms)
## ======================================================================


cmp = (
    df_wide[
        df_wide["simulation"] == "CMP"
    ]
)

cmpd = (
    df_wide[
        df_wide["simulation"] == "CMPD"
    ]
)

merge_cols = [
    "parameter_set",
    "scenario",
    "time",
    "experiment_type"
]

df_delta = cmp.merge(
    cmpd,
    on=merge_cols,
    suffixes=("_CMP", "_CMPD"),
    validate="one_to_one"
)


pairs = [
    "C-M",
    "C-P",
    "M-P"
]

for pair in pairs:

    df_delta[f"delta_{pair}_WGT"] = (
        df_delta[f"{pair}_WGT_CMPD"]
        -
        df_delta[f"{pair}_WGT_CMP"]
    )


for pair in pairs:

    df_delta[f"delta_{pair}_BC"] = (
        df_delta[f"{pair}_BC_CMPD"]
        -
        df_delta[f"{pair}_BC_CMP"]
    )

cols = [
    c
    for c in df_delta.columns
    if "delta_" in c
]

print(df_delta[cols].describe())

print(
    df_delta[
        [
            "delta_C-P_WGT",
            "C-P_WGT_CMP",
            "C-P_WGT_CMPD"
        ]
    ].describe()
)

pairs = [
    "C-M",
    "C-P",
    "M-P"
]

fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(15,5),
    sharey=True
)

for ax, pair in zip(axes, pairs):

    for exp_type, color in [
        ("baseline", "black"),
        ("alternative", "tab:orange")
    ]:

        sub = df_delta[
            df_delta["experiment_type"] == exp_type
        ]

        summary = (
            sub.groupby("time")[f"delta_{pair}_BC"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.plot(
            summary["time"],
            summary["mean"],
            color=color,
            label=exp_type,
            linewidth=2
        )

        ax.fill_between(
            summary["time"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=color,
            alpha=0.2
        )

    ax.axhline(0, color="grey", ls="--")
    ax.set_title(pair)

axes[-1].legend()

plt.tight_layout()

plt.show()




## ======================================================================
## FIGURE 6: Différence relative pour chaque espèce en fonction du temps
## Mais différence entre coefficients de base seulement
## ======================================================================

distribution_results = []


parameter_sets = (
    df_wide["parameter_set"]
    .unique()
)


for pset in parameter_sets:

    cmp_paths = [
        d["path"]
        for d in simulation_sets
        if d["parameter_set"] == pset
        and d["simulation"] == "CMP"
    ]

    cmpd_paths = [
        d["path"]
        for d in simulation_sets
        if d["parameter_set"] == pset
        and d["simulation"] == "CMPD"
    ]

    if len(cmp_paths) == 0:
        continue

    if len(cmpd_paths) == 0:
        print(f"Skip {pset}: no CMPD")
        continue

    cmp_path = cmp_paths[0]
    cmpd_path = cmpd_paths[0]


    cmp_scenarios = sorted(
        glob.glob(
            os.path.join(
                cmp_path,
                "scenario_*"
            )
        )
    )

    cmpd_scenarios = sorted(
        glob.glob(
            os.path.join(
                cmpd_path,
                "scenario_*"
            )
        )
    )



    for cmp_dir, cmpd_dir in zip(
        cmp_scenarios,
        cmpd_scenarios
    ):

        scenario = os.path.basename(
            cmp_dir
        )

        cmp_files = sorted(
            glob.glob(
                os.path.join(
                    cmp_dir,
                    "*.cont.vtu"
                )
            ),
            key=extract_time
        )

        cmpd_files = sorted(
            glob.glob(
                os.path.join(
                    cmpd_dir,
                    "*.cont.vtu"
                )
            ),
            key=extract_time
        )

        for f_cmp, f_cmpd in zip(
            cmp_files,
            cmpd_files
        ):

            t = extract_time(
                f_cmp
            )

            cmp_data, areas = (
                get_species_from_vtu(
                    f_cmp
                )
            )

            cmpd_data, _ = (
                get_species_from_vtu(
                    f_cmpd
                )
            )


            delta_C = (
                relative_l1_difference(
                    cmp_data["C"],
                    cmpd_data["C"],
                    areas
                )
            )

            delta_M = (
                relative_l1_difference(
                    cmp_data["M"],
                    cmpd_data["M"],
                    areas
                )
            )

            delta_P = (
                relative_l1_difference(
                    cmp_data["P"],
                    cmpd_data["P"],
                    areas
                )
            )


            cp_cmp = (
                cmp_data["C"]
                * cmp_data["P"]
            )

            cp_cmpd = (
                cmpd_data["C"]
                * cmpd_data["P"]
            )

            delta_CP = (
                relative_l1_difference(
                    cp_cmp,
                    cp_cmpd,
                    areas
                )
            )

            cm_cmp = (
                cmp_data["C"]
                * cmp_data["M"]
            )

            cm_cmpd = (
                cmpd_data["C"]
                * cmpd_data["M"]
            )

            delta_CM = (
                relative_l1_difference(
                    cm_cmp,
                    cm_cmpd,
                    areas
                )
            )


            distribution_results.append(
                {
                    "parameter_set": pset,
                    "experiment_type": (
                        "baseline"
                        if pset == "i"
                        else "alternative"
                    ),
                    "scenario": scenario,
                    "time": t,

                    "delta_C": delta_C,
                    "delta_M": delta_M,
                    "delta_P": delta_P,

                    "delta_CP": delta_CP,
                    "delta_CM": delta_CM
                }
            )


df_dist = pd.DataFrame(
distribution_results
)

print(
    df_dist.describe()
)

fig, axes = plt.subplots(
    1,
    5,
    figsize=(25,5)
)

variables = [
    "delta_C",
    "delta_M",
    "delta_P",
    "delta_CP",
    "delta_CM"
]

for exp_type, sub in df_dist.groupby("experiment_type"):

    tmp = (
        sub.groupby("time")[variables]
        .mean()
        .reset_index()
    )

    ax.plot(
        tmp["time"],
        tmp[variables],
        label=exp_type
    )

for ax, variables in zip(
    axes,
    variables
):

    tmp = (
        df_dist
        .groupby("time")[variables]
        .mean()
        .reset_index()
    )

    ax.plot(
        tmp["time"],
        tmp[variables],
        lw=3
    )

    ax.set_title(variables)

    ax.set_xlabel(
        "Temps"
    )

axes[0].set_ylabel(
    "Différence relative"
)

plt.tight_layout()
plt.show()

print(
    df_dist[
        [
            "delta_C",
            "delta_M",
            "delta_P",
            "delta_CP",
            "delta_CM"
        ]
    ].describe()
)


## ======================================================================
## FIGURE 7: pour chaque espèce, calcul de son alignement avec ses
## ressources dans le temps, selon scenario X parameter sets
## ======================================================================


for col in df_wide.columns:
    if "ALIGN" in col:
        print(col)



color_map = {
    ("CMP", "baseline"): "tab:blue",
    ("CMP", "alternative"): "tab:cyan",
    ("CMPD", "baseline"): "tab:red",
}


alignments = [
    a for a in [
        "C-Phi_C_ALIGN",
        "M-Phi_M_ALIGN",
        "P-Phi_P_ALIGN",
        "D-Phi_D_ALIGN"
    ]
    if a in df_wide.columns
]


print("Alignments trouvés :", alignments)

fig, axes = plt.subplots(
    1,
    len(alignments),
    figsize=(5*len(alignments), 5),
    sharey=True
)

if len(alignments) == 1:
    axes = [axes]

legend_handles = {}

for ax, var in zip(axes, alignments):

    for (sim, exp_type), sub_all in (
    df_wide.groupby(
        ["simulation", "experiment_type"]
    )
):

        for scen, sub in (
            sub_all.groupby("scenario")
        ):

            sub = sub.sort_values("time")

            line = ax.plot(
                sub["time"],
                sub[var],
                alpha=1,
                linewidth=1,
                color=color_map[(sim, exp_type)]
            )[0]

            ax.text(
                sub["time"].iloc[-1],
                sub[var].iloc[-1],
                sub["SC_label"].iloc[0],
                fontsize=6
            )

            key = f"{sim} - {exp_type}"

            if key not in legend_handles:
                legend_handles[key] = line

    ax.set_title(
        var.replace("_ALIGN", "")
    )

    ax.set_xlabel("Temps")

axes[0].set_ylabel(
    "Alignement ressource-potentiel"
)

axes[-1].legend(
    legend_handles.values(),
    legend_handles.keys(),
    title="Simulation"
)

plt.tight_layout()
plt.show()


## ======================================================================
## FIGURE 8: ratio entre l'overlap avec la ressource et l'overlap 
## avec le prédateur en fct du temps
## ======================================================================

# fig, ax = plt.subplots(figsize=(8,5))

# for (pset, sim, scen), sub in (
#     df_wide.groupby(
#         ["parameter_set","simulation", "scenario"]
#     )
# ):

#     sub = sub.sort_values("time")

#     ax.plot(
#         sub["time"],
#         sub["C_ratio_P"],
#         label=f"{pset}-{sim}-{scen}:P"
#     )

#     ax.plot(
#         sub["time"],
#         sub["C_ratio_PM"],
#         "--",
#         label=f"{pset}-{sim}-{scen}: P+M"
#     )

#     ax.plot(
#         sub["time"],
#         sub["C_ratio_PMD"],
#         ":",
#         label=f"{pset}-{sim}-{scen}: P+M+D"
#     )

# ax.legend()

# ax.set_xlabel("Time")
# ax.set_ylabel("Overlap entre la distribution du caribou et le paysage de la peur")

# plt.show()


## 2eme proposition pour le meme graphique

## Couleur: CMP (bleu) et CMPD (rouge)
## Type de trait: quelles especes on utilise pour computer le paysage de la peur: P, PM ou PMD
## Epaisseur du trait: + épais si scenario de baseline, + fin si alternatif.
## Mais il faut trouver uen facon de dire en quoi ces scenarios sont alternatifs?

fig, ax = plt.subplots(figsize=(8,5))

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="tab:blue", lw=2, label="CMP"),
    Line2D([0], [0], color="tab:red", lw=2, label="CMPD"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="P"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="P+M"),
    Line2D([0], [0], color="black", lw=2, linestyle=":", label="P+M+D")
]

colors = {
    "CMP": "tab:blue",
    "CMPD": "tab:red"
}

linestyles = {
    "P": "-",
    "PM": "--",
    "PMD": ":"
}


for (pset, sim, scen), sub in df_wide.groupby(
    ["parameter_set", "simulation", "scenario"]
):

    sub = sub.sort_values("time")

    lw = (
        3 if pset == "i"
        else 1)
    
    ax.plot(
        sub["time"],
        sub["C_ratio_P"],
        color=colors[sim],
        linestyle=linestyles["P"],
        linewidth=lw
    )

    ax.plot(
        sub["time"],
        sub["C_ratio_PM"],
        color=colors[sim],
        linestyle=linestyles["PM"],
        linewidth=lw
    )

    ax.plot(
        sub["time"],
        sub["C_ratio_PMD"],
        color=colors[sim],
        linestyle=linestyles["PMD"],
        linewidth=lw
    )

    ax.text(
        sub["time"].iloc[-1],
        sub["C_ratio_P"].iloc[-1],
        sub["SC_label"].iloc[0],
        fontsize=6
    )  

ax.legend(handles=legend_elements)  

ax.set_xlabel("Time")
ax.set_ylabel("Overlap entre la distribution du caribou et le paysage de la peur")

plt.show()



## ======================================================================
## FIGURE 9: Évolution temporelle entre la distribution du caribou 
## et les différents paysages de risque
## ======================================================================
variables = [
    "C-riskP_ALIGN",
    "C-riskPM_ALIGN",
    "C-riskPMD_ALIGN"
]

titles = [
    "Risque P",
    "Risque P+M",
    "Risque P+M+D"
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharey=False
)

colors = {
    ("CMP", "baseline"): "tab:blue",
    ("CMP", "alternative"): "tab:cyan",
    ("CMPD", "baseline"): "tab:red",
    ("CMPD", "alternative"): "tab:orange",
}


for ax, var, title in zip(axes, variables, titles):

    for (sim, exp_type, scen), sub in (
    df_wide.groupby(
        [
            "simulation",
            "experiment_type",
            "scenario"
        ]
    )
):

        sub = sub.sort_values("time")

        ax.plot(
            sub["time"],
            sub[var],
            color=colors[(sim, exp_type)],
            linewidth=1
        )

        ax.text(
            sub["time"].iloc[-1],
            sub[var].iloc[-1],
            sub["SC_label"].iloc[0],
            fontsize=6
        )

    ax.set_title(title)
    ax.set_xlabel("Temps")
    ax.grid(alpha=0.3)

axes[0].set_ylabel(
    "Alignement distribution-risque"
)

# légende unique
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="tab:blue", label="CMP baseline"),
    Line2D([0], [0], color="tab:cyan", label="CMP alternatif"),
    Line2D([0], [0], color="tab:red", label="CMPD baseline"),
    Line2D([0], [0], color="tab:orange", label="CMPD alternatif"),
]

axes[-1].legend(handles=legend_elements)

plt.tight_layout()
plt.show()



## ======================================================================
## FIGURE 10: Ratio ressource / prédateur au temps final
## ======================================================================

tmax = df_wide["time"].max()

df_final = (
    df_wide[
        df_wide["time"] == tmax
    ]
)

ratios = [
    "C_ratio_P",
    "M_ratio_P",
    "D_ratio_P"
]

titles = [
    "Caribou",
    "Moose",
    "Deer"
]

fig, axes = plt.subplots(
    1,
    len(ratios),
    figsize=(15,5),
    sharey=False
)

for ax, ratio, title in zip(
    axes,
    ratios,
    titles
):

    for sim, marker, color in [
        ("CMP", "o", "tab:blue"),
        ("CMPD", "^", "tab:red")
    ]:

        sub = df_final[
        df_final["simulation"] == sim]

        ## Ajoute un peu de jitter horizontal si necessaire
        offset = {
            "CMP": -0.01,
            "CMPD": 0.01
            }

        ax.scatter(
            sub["c_ratio"],
            sub[ratio],
            marker=marker,
            color=color,
            s=100,
            alpha=0.8,
            label=sim
        )

        for _, row in sub.iterrows():

            ax.annotate(
                row["SC_label"],
                (
                    row["c_ratio"],
                    row[ratio]
                ),
                fontsize=8,
                xytext=(4,4),
                textcoords="offset points"
            )

        # ax.plot(
        #     sub["c_ratio"],
        #     sub[ratio],
        #     color=color,
        #     alpha=0.5,
        #     linewidth=1.5
        # )

    ax.set_title(title)

    ax.set_xlabel(
        r"$C_{IFD}/C_{LOF}$"
    )

    ax.grid(
        alpha=0.3
    )

axes[0].set_ylabel(
    "Ressource / Prédateur"
)

axes[-1].legend()

plt.tight_layout()
plt.show()



## ======================================================================
## FIGURE 12: Mesure de l'expansion spatiale en fct du temps
## On considère une cellule occupée si la densité de l,espèce dedans est
## supérieure a un seuil de 1% de sa densité initiale
## et calcule l'aire occupée
## ======================================================================

df_spread = pd.DataFrame(
    spread_results
)

print(df_spread.head())

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10,8),
    sharex=True
)

species_list = ["C","M","P","D"]

for ax, sp in zip(
    axes.ravel(),
    species_list
):

    for sim in ["CMP","CMPD"]:

        sub = (
            df_spread[
                (df_spread["species"] == sp)
                &
                (df_spread["simulation"] == sim)
            ]
            .groupby("time")["spread"]
            .mean()
            .reset_index()
        )

        ax.plot(
            sub["time"],
            sub["spread"],
            label=sim
        )

        # ax.text(
        #     sub["time"].iloc[-1],
        #     sub["spread"].iloc[-1],
        #     sub["SC_label"].iloc[0],
        #     fontsize=6
        # )

    ax.set_title(sp)

for ax in axes[1,:]:
    ax.set_xlabel("Temps")

for ax in axes[:,0]:
    ax.set_ylabel("Fraction du domaine occupée")

axes[0,0].legend()

plt.tight_layout()
plt.show()



## ======================================================================
## FIGURE 13: Time to reach equilibrium ou REGARDER LA VITESSE DE CONVERGENCE
## ======================================================================


def detect_equilibrium_time(
    spread_series,
    time_series,
    epsilon=1e-4
):
    """
    Trouve le premier temps à partir duquel
    la variation du spread reste inférieure à epsilon.
    """

    spread = spread_series.values

    rel_change = np.abs(
        np.diff(spread)
    )

    for i in range(len(rel_change)):

        if np.all(
            rel_change[i:] < epsilon
        ):
            return time_series.iloc[i+1]

    return np.nan


equilibrium_results = []


threshold = 0.95

records = []

pairs = [
    "C-M",
    "C-P",
    "C-D",
    "M-P",
    "M-D"
]

for pair in pairs:

    tmp = df[
    df["pair"] == pair
    ]

    for keys, sub in tmp.groupby(
        [
            "scenario",
            "simulation",
            "experiment_type"
        ]
    ):

        sub = sub.sort_values("time")

        reached = sub.loc[
            sub["BC"] >= threshold,
            "time"
        ]

        t95 = (
            reached.iloc[0]
            if len(reached) > 0
            else np.nan
        )

        records.append(
            dict(
                zip(
                    [
                        "scenario",
                        "simulation",
                        "experiment_type"
                    ],
                    keys
                ),
                pair=pair,
                T95=t95
            )
        )

df_t95 = pd.DataFrame(records)



# ================ Tentative de semi-variogramme ? Comme dans Farnworth1999?