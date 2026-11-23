import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


## ==============================================================================
## ==============================================================================
## FIGURES POUR OBJECTIF i)
# Ce script charge les resultats de simulations de baseline
# et fait les figures à partir d'eux
## ==============================================================================
## ==============================================================================


# ====================== LECTURE DES DONNEES ======================
df = pd.read_pickle(
    "outputs/Bhattacharyya_long.pkl"
)

df_wide = pd.read_pickle(
    "outputs/Bhattacharyya_wide.pkl"
)

df_spread = pd.read_pickle(
    "outputs/spread_results.pkl"
)

df_dist = pd.read_pickle(
    "outputs/distribution_results.pkl"
)


# ==================== CREATION DU STYLE POUR LES FIGURES =============

df_wide = pd.read_pickle(
    "outputs/Bhattacharyya_wide.pkl"
)

df_wide["SC_label"] = (
    df_wide["simulation"]
    + "_"
    + df_wide["SC"]
)

df_wide["C_IFD_diff_P"] = (
    df_wide["C-IFDresource_ALIGN"]
    - df_wide["C-riskP_ALIGN"]
)

df_wide["C_IFD_diff_PM"] = (
    df_wide["C-IFDresource_ALIGN"]
    - df_wide["C-riskPM_ALIGN"]
)

df_wide["C_IFD_diff_PMD"] = (
    df_wide["C-IFDresource_ALIGN"]
    - df_wide["C-riskPMD_ALIGN"]
)


df_wide["M_IFD_diff_P"] = (
    df_wide["M-IFDresource_ALIGN"]
    - df_wide["M-riskP_ALIGN"]
)
df_wide["M_IFD_diff_PC"] = (
    df_wide["M-IFDresource_ALIGN"]
    - df_wide["M-riskPC_ALIGN"]
)
df_wide["M_IFD_diff_PCD"] = (
    df_wide["M-IFDresource_ALIGN"]
    - df_wide["M-riskPCD_ALIGN"]
)


df_wide["D_IFD_diff_P"] = (
    df_wide["D-IFDresource_ALIGN"]
    - df_wide["D-riskP_ALIGN"]
)




df_spread["SC"] = (
    "SC"
    + df_spread["scenario"]
        .str.replace("scenario_", "", regex=False)
        .astype(int)
        .astype(str)
)


print(
    df_wide[
        ["scenario","SC","experiment_type"]
    ]
    .drop_duplicates()
    .sort_values("SC")
)

unique_sc = sorted(
    df_wide["SC"].unique()
)


## On definit les couleurs pour chacuns des scenarios
scenario_colors = {
    "BASELINE": "black",
    "SC0": "#1F77B4",  # bleu
    "SC1": "#FF7F0E",  # orange
    "SC2": "#2CA02C",  # vert
    "SC3": "#D62728",  # rouge
    "SC4": "#9467BD",  # violet
    "SC5": "#8C564B",  # brun
    "SC6": "#E377C2",  # rose
    "SC7": "#7F7F7F",  # gris
    "SC8": "#BCBD22",  # olive
    "SC9": "#17BECF",  # cyan
}




experiment_styles = {
    "baseline": {
        "linestyle": "-",
        "lw": 3,
        "alpha": 1.0,
        "color": "black"
    },

    "ii_native_naive": {
        "linestyle": "--",
        "lw": 2,
        "alpha": 1.0,
        "color": "tab:blue"
    },

    "ii_native_non_naive": {
        "linestyle": ":",
        "lw": 2,
        "alpha": 1.0,
        "color": "tab:orange"
    }
}


all_scenarios = sorted(df_wide["SC"].unique())


baseline = df_wide[
    df_wide["experiment_type"] == "baseline"
]

baseline_cmp = baseline[
    baseline["simulation"] == "CMP"
]

baseline_cmpd = baseline[
    baseline["simulation"] == "CMPD"
]





#       Comparaison du chevauchement entre paires d'especes entre CMP et CMPD 
#       3 subplots: Caribou-Moose ; Caribou-Wolf ; Moose-Wolf
#       On calcule le BC pour chaque paire, chaque scenario, dans le temps 
#       (6 groupes de données)
#       puis on calcule le Delta_BC = BC(CMPD) - BC(CMP)
#       Valeurs négatives: Delta_BC < 0; donc BC(CMPD) < BC(CMP)
#       Valeurs positives: Delta_BC > 0; donc BC(CMPD) > BC(CMP)
# ==========================================================================================

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
    "M-P",
    "C-D",
    "M-D",
    "P-D"
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



pair_titles = {
    "C-M": "Caribou-Moose",
    "C-P": "Caribou-Wolf",
    "M-P": "Moose-Wolf",
    "C-D": "Caribou-Deer",
    "M-D": "Moose-Deer",
    "P-D": "Wolf-Deer"
}


pairs_native = [
    "C-M",
    "C-P",
    "M-P"
]

pairs_deer = [
    "C-D",
    "M-D",
    "P-D"
]

pairs_all = (
    pairs_native
    + pairs_deer
)


fig, axes = plt.subplots(
    1,
    len(pairs_native),
    figsize=(15,5),
    sharey=True
)

for ax, pair in zip(axes, pairs_native):

    for exp_type, color in [

    ("baseline", "black")

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
 
    ax.set_title(
    pair_titles[pair],
    fontsize=13,
    fontweight="bold"
)

    fig.supxlabel(
    "Time",
    fontsize=12
)
    
fig.supylabel(
    r"$\Delta BC$ (CMPD - CMP)",
    fontsize=12
)

plt.tight_layout()

plt.show()



## ==============================================================================
# FIGURE 2:
#       Effet de l'invasion du cerf en comparant CMPD - CMP
#       2 colonnes: Caribou / Orignal
#       3 lignes: 
#       Δ Alignement ressource (M1 + M2)
#       Δ Alignement prédation: P (trait plein), P+M (tirets), P+M+D (pointillés)
#       Δ Alignement ressource - prédation: P (trait plein), P+M (tirets), P+M+D (pointillés)
## ==============================================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(15, 15),
    sharex=True,
    sharey="row"
)

# ------------------------------------------------------------------
# Titres des colonnes
# ------------------------------------------------------------------

col_titles = [
    "Caribou",
    "Moose"
]

for j, title in enumerate(col_titles):

    axes[0, j].set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )


# ------------------------------------------------------------------
# Ligne 1 : Δ Alignement ressource
# ------------------------------------------------------------------

# resource_info = [
#     "C-Phi_C_ALIGN",
#     "M-Phi_M_ALIGN",
#     "C-Phi_C_ALIGN",
# ]


resource_info = [
        "C-IFDresource_ALIGN",
        "M-IFDresource_ALIGN"
]



for j, metric in enumerate(resource_info):

    delta = (
        baseline_cmpd[metric].values
        - baseline_cmp[metric].values
    )

    ax = axes[0, j]

    ax.plot(
        baseline_cmp["time"],
        delta,
        color="tab:blue",
        lw=2
    )

    ax.axhline(
        0,
        color="grey",
        ls="--",
        lw=1
    )

    ax.axhline(
        0,
        color="grey",
        ls="--",
        lw=1
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )


# ------------------------------------------------------------------
# Ligne 2 : Δ Alignement risque de prédation
# ------------------------------------------------------------------

pred_info = {
    0: {  # Caribou-Prédateur
        "P": ("C-riskP_ALIGN", "-"),
        "P+M": ("C-riskPM_ALIGN", "--"),
        "P+M+D": ("C-riskPMD_ALIGN", ":")
    },

    1: {  # Orignal-Prédateur
        "P": ("M-riskP_ALIGN", "-"),
        "P+C": ("M-riskPC_ALIGN", "--"),
        "P+C+D": ("M-riskPCD_ALIGN", ":")
    }
}

colors = {
    "P": "firebrick",
    "P+M": "darkorange",
    "P+M+D": "purple",
    "P+C": "darkorange",
    "P+C+D": "purple"
}

for j in [0, 1]:

    ax = axes[1, j]

    for label, (metric, ls) in pred_info[j].items():

        delta = (
            baseline_cmpd[metric].values
            - baseline_cmp[metric].values
        )

        ax.plot(
            baseline_cmp["time"],
            delta,
            color=colors[label],
            linestyle=ls,
            linewidth=2,
            label=label
        )

    ax.axhline(
        0,
        color="grey",
        linestyle="--",
        linewidth=1
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        title="Risk landscape"
    )


# ------------------------------------------------------------------
# Ligne 3 : Δ Resource/Risk ratio
# ------------------------------------------------------------------
## Ancien code qui utilisait les ratios instead of differences
# ratio_info = {
#     0: {
#         "P": ("C_IFD_ratio_P", "-"),
#         "P+M": ("C_IFD_ratio_PM", "--"),
#         "P+M+D": ("C_IFD_ratio_PMD", ":")
#     },

#     1: {
#         "P": ("M_IFD_ratio_P", "-"),
#         "P+C": ("M_IFD_ratio_PC", "--"),
#         "P+C+D": ("M_IFD_ratio_PCD", ":")
#     }
# }


## En utilisant les différences alignement ressource - alignement prédateur
# Si le ratio est > 0: l'espece est davantage alignée avec la ressource qu'avec le risque
# Si le ratio est < 0: l'espece est davantage alignée avec le risque qu'avec la ressource

ratio_info = {
    0: {
        "P": ("C_IFD_diff_P", "-"),
        "P+M": ("C_IFD_diff_PM", "--"),
        "P+M+D": ("C_IFD_diff_PMD", ":")
    },

    1: {
        "P": ("M_IFD_diff_P", "-"),
        "P+C": ("M_IFD_diff_PC", "--"),
        "P+C+D": ("M_IFD_diff_PCD", ":")
    }
}


for j in [0, 1]:

    ax = axes[2, j]

    for label, (metric, ls) in ratio_info[j].items():

        delta = (
            baseline_cmpd[metric].values
            - baseline_cmp[metric].values
        )

        ax.plot(
            baseline_cmp["time"],
            delta,
            color=colors[label],
            linestyle=ls,
            linewidth=2,
            label=label
        )

    ax.axhline(
        0,
        color="grey",
        linestyle="--",
        linewidth=1
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        title="Risk landscape"
    )



# ------------------------------------------------------------------
# Labels des lignes
# ------------------------------------------------------------------

axes[0,0].set_ylabel(
    r"$\Delta$ Resource alignment",
    fontsize=12,
    color="tab:blue"
)

axes[1,0].set_ylabel(
    r"$\Delta$ Predation-risk alignment",
    fontsize=12,
    color="tab:red"
)

axes[2,0].set_ylabel(
    r"$\Delta$ (Resource - Risk) alignment",
    fontsize=12,
    color="black"
)

# ------------------------------------------------------------------
# Labels x
# ------------------------------------------------------------------

for ax in axes[2, :]:
    ax.set_xlabel(
        "Time",
        fontsize=11
    )


plt.tight_layout()
plt.show()



# ==========================================================================================
# FIGURE 3:
        # Spread rate through time 
        # Un panneau pour chacune des espèces, ou on mesure la proportion de cellules
        # occupées dans le domaine (densité > valeur seuil)
# ==========================================================================================

# ============================
# STYLE GENERAL
# ============================

species_list = ["C", "M", "P", "D"]

col_titles = [
    "Caribou",
    "Moose",
    "Wolf",
    "Deer",
]

simulation_colors = {
    "CMP": "tab:blue",
    "CMPD": "tab:red",
}

# ============================
# CREATION DE LA FIGURE
# ============================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharex=True
)

for ax, sp, title in zip(
    axes.ravel(),
    species_list,
    col_titles
):

    for sim in ["CMP", "CMPD"]:

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
            color=simulation_colors[sim],
            linewidth=2,
            label=sim,
        )

    # Titre par subplot
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    # Grille
    ax.grid(
        linestyle=":",
        alpha=0.3
    )

# Axe x
for ax in axes[1, :]:
    ax.set_xlabel("Time")

# Axe y
for ax in axes[:, 0]:
    ax.set_ylabel(
        "Proportion of occupied domain"
    )

# Légende unique
handles = [
    Line2D(
        [0], [0],
        color="tab:blue",
        lw=2,
        label="CMP"
    ),
    Line2D(
        [0], [0],
        color="tab:red",
        lw=2,
        label="CMPD"
    )
]

fig.subplots_adjust(right=0.82)

fig.legend(
    handles=handles,
    title="Simulation",
    loc="center left",
    bbox_to_anchor=(0.84, 0.5),
    frameon=False,
)

plt.show()


# ==========================================================================================
# FIGURE 4:
        # Temps final, on compare les ratios alignement ressource/predateur
        # pour les 3 espèces de proies
        # selon si on regarde l'alignement avec les ressources M1 + M2
        # ou l'alignement avec le potentiel Phi
# ==========================================================================================

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


species_info = {
    "C": {
        "ifd_ratio": "C_IFD_ratio_P",
        "phi_ratio": "C_PHI_ratio_P",
        "x": "c_ratio",
        "label": "Caribou",
    },
    "M": {
        "ifd_ratio": "M_IFD_ratio_P",
        "phi_ratio": "M_PHI_ratio_P",
        "x": "m_ratio",
        "label": "Moose",
    },
    "D": {
        "ifd_ratio": "D_IFD_ratio_P",
        "phi_ratio": "D_PHI_ratio_P",
        "x": "d_ratio",
        "label": "Deer",
    },
}


## Filtrage des datas au temps final
tmax = df_wide["time"].max()

df_final = df_wide[
    (df_wide["time"] == tmax)
    & (df_wide["experiment_type"] == "baseline")
].copy()


# ============================
# STYLE GENERAL
# ============================

# Couleurs des simulations
simulation_colors = {
    "CMP": "tab:blue",
    "CMPD": "tab:red",
}

# Formes des métriques
metric_markers = {
    "IFD": "o",
    "PHI": "s",
}

# ============================
# CREATION DE LA FIGURE
# ============================


fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 5),
    sharex=True,
    sharey=True,
)

for ax, (sp, info) in zip(axes, species_info.items()):

    for sim in ["CMP", "CMPD"]:

        sub = df_final[
            df_final["simulation"] == sim
        ]

        # Ratio basé sur IFD
        ax.scatter(
            sub[info["x"]],
            sub[info["ifd_ratio"]],
            color=simulation_colors[sim],
            marker=metric_markers["IFD"],
            s=120,
            edgecolors="black",
            alpha=0.8,
        )

        # Ratio basé sur PHI
        ax.scatter(
            sub[info["x"]],
            sub[info["phi_ratio"]],
            color=simulation_colors[sim],
            marker=metric_markers["PHI"],
            s=120,
            edgecolors="black",
            alpha=0.8,
        )

    # Titres
    ax.set_title(
        info["label"],
        fontsize=13,
        fontweight="bold",
    )

    # Zoom uniquement pour le Moose
    if sp == "M":

        axins = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc="lower right"
        )

        for sim in ["CMP", "CMPD"]:

            sub = df_final[
                df_final["simulation"] == sim
            ]

            axins.scatter(
                sub[info["x"]],
                sub[info["ifd_ratio"]],
                color=simulation_colors[sim],
                marker=metric_markers["IFD"],
                s=50,
                edgecolors="black",
                alpha=0.8,
            )

            axins.scatter(
                sub[info["x"]],
                sub[info["phi_ratio"]],
                color=simulation_colors[sim],
                marker=metric_markers["PHI"],
                s=50,
                edgecolors="black",
                alpha=0.8,
            )

        # Limites du zoom
        axins.set_ylim(-0.75, -0.65)

        # Ajuste automatiquement les x selon les données du Moose
        xvals = np.concatenate([
            sub[info["x"]].values
            for sim in ["CMP", "CMPD"]
            for _, sub in [(
                sim,
                df_final[df_final["simulation"] == sim]
            )]
        ])

        axins.set_xlim(
            xvals.min() - 0.05,
            xvals.max() + 0.05
        )

        axins.grid(
            linestyle=":",
            alpha=0.3
        )

        axins.tick_params(
            labelsize=7
        )

    # Label de l'axe x
    ax.set_xlabel(r"$IFD/LOF$")

    # Grille
    ax.grid(
        linestyle=":",
        alpha=0.3,
    )

# Axe Y commun
axes[0].set_ylabel(
    "Resource alignment / Predator alignment"
)


# ============================
# LEGENDE 1 : SIMULATIONS (couleur)
# ============================

simulation_legend = [
    Line2D(
        [0], [0],
        marker="o",
        color="tab:blue",
        linestyle="",
        markeredgecolor="black",
        markersize=10,
        label="CMP",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="tab:red",
        linestyle="",
        markeredgecolor="black",
        markersize=10,
        label="CMPD",
    ),
]

# ============================
# LEGENDE 2 : METRIQUES (forme)
# ============================
metric_legend = [
    Line2D(
        [0], [0],
        marker="o",
        color="black",
        linestyle="",
        markersize=10,
        label="M1+M2",
    ),
    Line2D(
        [0], [0],
        marker="s",
        color="black",
        linestyle="",
        markersize=10,
        label=r"$\Phi$",
    ),
]

# Placement des légendes
leg1 = fig.legend(
    handles=simulation_legend,
    title="Simulation",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.95),
    frameon=False,
)

leg2 = fig.legend(
    handles=metric_legend,
    title="Resource metric",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.72),
    frameon=False,
)

fig.add_artist(leg1)
fig.add_artist(leg2)


# ============================
# MISE EN PAGE
# ============================

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.show()


print(
    df_final.groupby("simulation")[
        [
            "C_PHI_ratio_P",
            "C_IFD_ratio_P"
        ]
    ].mean()
)





# ==========================================================================================
# FIGURE 5:
# Trajectoires Resources alignment vs Predation-risk alignment
# CMP (rouge) vs CMPD (bleu)
# ==========================================================================================

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Baseline seulement
# ============================================================

baseline = df_wide[
    df_wide["experiment_type"] == "baseline"
].copy()

# ============================================================
# Espèces
# ============================================================

species_info = {
    "Caribou": (
        "C-IFDresource_ALIGN",
        "C-riskPMD_ALIGN",
    ),
    "Moose": (
        "M-IFDresource_ALIGN",
        "M-riskPCD_ALIGN",
    ),
    "Deer": (
        "D-IFDresource_ALIGN",
        "D-riskPCM_ALIGN",
    )
}

# ============================================================
# Couleurs simulations
# ============================================================

simulation_colors = {
    "CMP": "tab:red",
    "CMPD": "tab:blue"
}

# ============================================================
# Temps sélectionnés
# ============================================================

times = np.sort(
    baseline["time"].unique()
)

selected_times = np.unique(
    times[
        np.linspace(
            0,
            len(times) - 1,
            5,
            dtype=int
        )
    ]
)

sizes = np.linspace(
    50,
    180,
    len(selected_times)
)

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 5),
    sharex=True,
    sharey=True
)

for ax, (sp, (res_var, risk_var)) in zip(
    axes,
    species_info.items()
):

    for sim in ["CMP", "CMPD"]:

        sub = (
            baseline[
                (baseline["simulation"] == sim)
                &
                (baseline["time"].isin(selected_times))
            ]
            .sort_values("time")
        )

        # --------------------------------------------------
        # Trajectoire
        # --------------------------------------------------

        ax.plot(
            sub[risk_var],
            sub[res_var],
            color=simulation_colors[sim],
            lw=2.5,
            alpha=0.9
        )

        # --------------------------------------------------
        # Points temporels
        # --------------------------------------------------

        for k in range(len(sub)):

            ax.scatter(
            sub[risk_var].iloc[k],
            sub[res_var].iloc[k],
            s=sizes[k],
            color=simulation_colors[sim],
            edgecolors="black",
            linewidth=0.8,
            alpha=0.65,
            zorder=3
        )

    ax.set_title(
        sp,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predation-risk alignment"
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

# ============================================================
# Axe Y
# ============================================================

axes[0].set_ylabel(
    "Resource alignment"
)

# ============================================================
# Légende simulation
# ============================================================

simulation_legend = [

    Line2D(
        [0], [0],
        color="tab:red",
        lw=3,
        label="CMP"
    ),

    Line2D(
        [0], [0],
        color="tab:blue",
        lw=3,
        label="CMPD"
    )
]

# ============================================================
# Légende temps
# ============================================================

time_legend = []

for t, s in zip(
    selected_times,
    sizes
):

    time_legend.append(
        plt.scatter(
            [],
            [],
            s=s,
            color="lightgrey",
            edgecolors="black",
            label=f"{int(t)}"
        )
    )

# Légende des scenarios (couleur)
fig.legend(
    handles=simulation_legend,
    title="Simulation",
    loc="upper left",
    bbox_to_anchor=(0.84, 0.95),
    frameon=False
)

# Légende du temps (taille)
fig.legend(
    handles=time_legend,
    title="Time",
    loc="center left",
    bbox_to_anchor=(0.84, 0.45),
    frameon=False
)

plt.tight_layout(
    rect=[0, 0, 0.82, 0.95]
)

plt.show()