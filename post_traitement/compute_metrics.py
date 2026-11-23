# ============================================================
# IMPORTS
# ============================================================

import os
import re
import glob
import itertools

import vtk
import numpy as np
import pandas as pd

from vtk.util.numpy_support import vtk_to_numpy



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



def compute_bc_matrix(
    cell_data,
    areas
):

    species = []

    for name in species_names:

        arr = vtk_to_numpy(
            cell_data.GetArray(name)
        )

        species.append(arr)

    results = {}

    for i, j in itertools.combinations(
        range(len(species_names)),
        2
    ):

        pair = (
            f"{species_names[i]}"
            f"-"
            f"{species_names[j]}"
        )

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



def compute_resource_alignment(
    cell_data,
    areas
):

    results = []

    # Alignements individuels
    for sp, resources in resource_map.items():

        sp_array = vtk_to_numpy(
            cell_data.GetArray(sp)
        )

        for resource in resources:

            arr = cell_data.GetArray(resource)

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
                    "pair": f"{sp}-{resource}",
                    "ALIGN": align
                }
            )

    # -------------------------------------------------
    # Ressources IFD = M1 + M2
    # -------------------------------------------------

    for sp in ["C", "M", "D"]:

        sp_array = vtk_to_numpy(
            cell_data.GetArray(sp)
        )

        resource_ifd = (
            vtk_to_numpy(
                cell_data.GetArray(f"M1_{sp}")
            )
            +
            vtk_to_numpy(
                cell_data.GetArray(f"M2_{sp}")
            )
        )

        align_ifd = resource_alignment(
            sp_array,
            resource_ifd,
            areas
        )

        results.append(
            {
                "pair": f"{sp}-IFDresource",
                "ALIGN": align_ifd
            }
        )

    predator_pairs = {

        "C": "P",

        "M": "P",

        "D": "P"
    }

    risk_fields = {

        "C": {
            "riskP": ["P"],
            "riskPM": ["P", "M"],
            "riskPD": ["P", "D"],
            "riskPMD": ["P", "M", "D"]
        },

        "M": {
            "riskP": ["P"],
            "riskPC": ["P", "C"],
            "riskPD": ["P", "D"],
            "riskPCD": ["P", "C", "D"]
        },

        "D": {
            "riskP": ["P"],
            "riskPM": ["P", "M"],
            "riskPC": ["P", "C"],
            "riskPCM": ["P", "C", "M"]
        },

        "P": {
            "riskP":["P"]
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
                "pair":
                    f"{prey}-{predator}_risk",

                "ALIGN":
                    align_pred
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


# ======================= LECTURE DES VTUs ==============================================

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


            ## Cherche dans le sous-dossier ii) pour lequel on a fait varier les parms 
            elif experiment == "ii_native_naive":
                experiment_type = "ii_native_naive"

            elif experiment == "ii_native_non_naive":
                            experiment_type = "ii_native_non_naive"

            ## Cherche dans le sous-dossier iii) pour lequel on a fait varier les parms de la proie invasive
            elif experiment == "iii_var_invasive":
                experiment_type = "iii_var_invasive"


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


density_cache = []
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
#    if experiment_type == "alternative":
#        scenario_list = scenario_list[:2]
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

        d_ifd = (
            params.get("d1",0)
            + params.get("d2",0)
        )

        d_lof = (
            params.get("d3",0)
            + params.get("d4",0)
            + params.get("d5",0)
            + params.get("d6",0)
        )



        for file in files:

                t = extract_time(file)

                mesh = read_vtu(file)

                mesh = point_to_cell(mesh)

                areas = compute_areas(mesh)

                cell_data = mesh.GetCellData()

                density_cache.append(
                    {
                        "parameter_set": parameter_set,
                        "experiment_type": experiment_type,
                        "scenario": scenario_name,
                        "simulation": scen_type,
                        "time": t,

                        "areas": areas,

                        "C": vtk_to_numpy(
                            cell_data.GetArray("C")
                        ).copy(),

                        "M": vtk_to_numpy(
                            cell_data.GetArray("M")
                        ).copy(),

                        "P": vtk_to_numpy(
                            cell_data.GetArray("P")
                        ).copy(),

                        "D": vtk_to_numpy(
                            cell_data.GetArray("D")
                        ).copy()
                    }
                )

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





                BC = compute_bc_matrix(
                cell_data,
                areas
            )


                
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
                                    cell_data,
                                    areas
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
                            "p_lof": p_lof,

                            "d_ifd": d_ifd,
                            "d_lof": d_lof
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
                            "p_lof": p_lof,

                            "d_ifd": d_ifd,
                            "d_lof": d_lof
                        }
                    )


df_density = pd.DataFrame(
    density_cache
)

distribution_results = []

merge_cols = [
    "parameter_set",
    "scenario",
    "time"
]

cmp = (
    df_density[
        df_density["simulation"] == "CMP"
    ]
)

cmpd = (
    df_density[
        df_density["simulation"] == "CMPD"
    ]
)

df_pairs = cmp.merge(
    cmpd,
    on=merge_cols,
    suffixes=("_CMP", "_CMPD"),
    validate="one_to_one"
)

for _, row in df_pairs.iterrows():

    areas = row["areas_CMP"]

    delta_C = relative_l1_difference(
        row["C_CMP"],
        row["C_CMPD"],
        areas
    )

    delta_M = relative_l1_difference(
        row["M_CMP"],
        row["M_CMPD"],
        areas
    )

    delta_P = relative_l1_difference(
        row["P_CMP"],
        row["P_CMPD"],
        areas
    )

    cp_cmp = (
        row["C_CMP"]
        * row["P_CMP"]
    )

    cp_cmpd = (
        row["C_CMPD"]
        * row["P_CMPD"]
    )

    delta_CP = relative_l1_difference(
        cp_cmp,
        cp_cmpd,
        areas
    )

    cm_cmp = (
        row["C_CMP"]
        * row["M_CMP"]
    )

    cm_cmpd = (
        row["C_CMPD"]
        * row["M_CMPD"]
    )

    delta_CM = relative_l1_difference(
        cm_cmp,
        cm_cmpd,
        areas
    )

    distribution_results.append(
        {
            "parameter_set":
                row["parameter_set"],

            "experiment_type":
                row["experiment_type_CMP"],

            "scenario":
                row["scenario"],

            "time":
                row["time"],

            "delta_C":
                delta_C,

            "delta_M":
                delta_M,

            "delta_P":
                delta_P,

            "delta_CP":
                delta_CP,

            "delta_CM":
                delta_CM
        }
    )


# ================ CREATION DES FICHIERS DE SAUVEGARDE DES METRIQUES =========================

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
    "p_lof",
    "d_ifd",
    "d_lof"
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

# ====================================================
# Alignement ressources fondamentales / risque
# ====================================================

df_wide["C_IFD_ratio_P"] = (
    df_wide["C-IFDresource_ALIGN"]
    /
    df_wide["C-riskP_ALIGN"]
)

df_wide["M_IFD_ratio_P"] = (
    df_wide["M-IFDresource_ALIGN"]
    /
    df_wide["M-riskP_ALIGN"]
)

df_wide["D_IFD_ratio_P"] = (
    df_wide["D-IFDresource_ALIGN"]
    /
    df_wide["D-riskP_ALIGN"]
)

# Caribou
df_wide["C_IFD_ratio_PM"] = (
    df_wide["C-IFDresource_ALIGN"]
    /
    df_wide["C-riskPM_ALIGN"]
)

df_wide["C_IFD_ratio_PMD"] = (
    df_wide["C-IFDresource_ALIGN"]
    /
    df_wide["C-riskPMD_ALIGN"]
)

# Moose
df_wide["M_IFD_ratio_PC"] = (
    df_wide["M-IFDresource_ALIGN"]
    /
    df_wide["M-riskPC_ALIGN"]
)

df_wide["M_IFD_ratio_PCD"] = (
    df_wide["M-IFDresource_ALIGN"]
    /
    df_wide["M-riskPCD_ALIGN"]
)

# Deer
df_wide["D_IFD_ratio_PM"] = (
    df_wide["D-IFDresource_ALIGN"]
    /
    df_wide["D-riskPM_ALIGN"]
)

df_wide["D_IFD_ratio_PCM"] = (
    df_wide["D-IFDresource_ALIGN"]
    /
    df_wide["D-riskPCM_ALIGN"]
)


df_wide["c_ratio"] = np.where(
    df_wide["c_lof"] > 0,
    df_wide["c_ifd"] / df_wide["c_lof"],
    np.nan
)

df_wide["m_ratio"] = np.where(
    df_wide["m_lof"] > 0,
    df_wide["m_ifd"] / df_wide["m_lof"],
    np.nan
)

df_wide["d_ratio"] = np.where(
    df_wide["d_lof"] > 0,
    df_wide["d_ifd"] / df_wide["d_lof"],
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
df_wide["C_PHI_ratio_P"] = (
    df_wide["C-Phi_C_ALIGN"]
    /
    df_wide["C-riskP_ALIGN"]
)

df_wide["C_PHI_ratio_PM"] = (
    df_wide["C-Phi_C_ALIGN"]
    /
    df_wide["C-riskPM_ALIGN"]
)

df_wide["C_PHI_ratio_PMD"] = (
    df_wide["C-Phi_C_ALIGN"]
    /
    df_wide["C-riskPMD_ALIGN"]
)


### Orignal
df_wide["M_PHI_ratio_P"] = (
    df_wide["M-Phi_M_ALIGN"]
    /
    df_wide["M-riskP_ALIGN"]
)

df_wide["M_PHI_ratio_PC"] = (
    df_wide["M-Phi_M_ALIGN"]
    /
    df_wide["M-riskPC_ALIGN"]
)

df_wide["M_PHI_ratio_PCD"] = (
    df_wide["M-Phi_M_ALIGN"]
    /
    df_wide["M-riskPCD_ALIGN"]
)

### Cerf
df_wide["D_PHI_ratio_P"] = (
    df_wide["D-Phi_D_ALIGN"]
    /
    df_wide["D-riskP_ALIGN"]
)

df_wide["D_PHI_ratio_PM"] = (
    df_wide["D-Phi_D_ALIGN"]
    /
    df_wide["D-riskPM_ALIGN"]
)

df_wide["D_PHI_ratio_PMC"] = (
    df_wide["D-Phi_D_ALIGN"]
    /
    df_wide["D-riskPCM_ALIGN"]
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
    df_wide["experiment_type"] != "baseline"
]

baseline_cmp = baseline[
    baseline["simulation"] == "CMP"
]

baseline_cmpd = baseline[
    baseline["simulation"] == "CMPD"
]

alternative_cmp = alternative[
    alternative["simulation"] == "CMP"
]

alternative_cmpd = alternative[
    alternative["simulation"] == "CMPD"
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

#
# SC pour les expériences non-baseline
#
scenario_map = {
    scen: f"SC{int(scen.split('_')[1])}"
    for scen in (
        df_wide.loc[
            df_wide["experiment_type"] != "baseline",
            "scenario"
        ]
        .unique()
    )
}

df_wide["SC"] = np.nan

#
# scénarios expérimentaux
#
mask = (
    df_wide["experiment_type"]
    != "baseline"
)

df_wide.loc[mask, "SC"] = (
    df_wide.loc[mask, "scenario"]
    .map(scenario_map)
)

#
# baseline
#
df_wide.loc[
    ~mask,
    "SC"
] = "BASELINE"

print(df_wide["SC"].value_counts())

df_wide["SC_label"] = (
    df_wide["simulation"]
    + "_"
    + df_wide["SC"]
)

df_spread = pd.DataFrame(
    spread_results
)

df_dist = pd.DataFrame(
distribution_results
)

os.makedirs(
    "outputs",
    exist_ok=True
)

# =========================
# Sauvegarde pickle
# =========================

df.to_pickle(
    "outputs/Bhattacharyya_long.pkl"
)

df_wide.to_pickle(
    "outputs/Bhattacharyya_wide.pkl"
)

df_spread.to_pickle(
    "outputs/spread_results.pkl"
)

df_dist.to_pickle(
    "outputs/distribution_results.pkl"
)

print("Fichiers pickle sauvegardés.")