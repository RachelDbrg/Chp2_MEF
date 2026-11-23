import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


## ==============================================================================
## ==============================================================================
# FIGURES -- Objectif ii)
# Ce script charge les resultats de simulations de baseline et ii)
# et fait les figures à partir d'eux
## ==============================================================================
## ==============================================================================


# ----------------------------- LECTURE DES DONNEES -----------------------------
## Puis filtrage pour ne garder que baseline, et ii)

df = pd.read_pickle(
    "outputs/Bhattacharyya_long.pkl"
)
df = df[
    df["experiment_type"].isin(
        [
            "baseline",
            "ii_native_naive",
            "ii_native_non_naive"
        ]
    )
].copy()

df_wide = pd.read_pickle(
    "outputs/Bhattacharyya_wide.pkl"
)
df_wide["SC_label"] = (
    df_wide["simulation"]
    + "_"
    + df_wide["SC"]
)
## On filtre le df en ne prenant que les scenarios baseline et ii)
df_wide = df_wide[
    df_wide["experiment_type"].isin(
        [
            "baseline",
            "ii_native_naive",
            "ii_native_non_naive"
        ]
    )
].copy()



df_spread = pd.read_pickle(
    "outputs/spread_results.pkl"
)
df_spread = df_spread[
    df_spread["experiment_type"].isin(
        [
            "baseline",
            "ii_native_naive",
            "ii_native_non_naive"
        ]
    )
].copy()
df_spread["SC"] = (
    "SC"
    + df_spread["scenario"]
        .str.replace("scenario_", "", regex=False)
        .astype(int)
        .astype(str)
)



df_dist = pd.read_pickle(
    "outputs/distribution_results.pkl"
)
df_dist = df_dist[
    df_dist["experiment_type"].isin(
        [
            "baseline",
            "ii_native_naive",
            "ii_native_non_naive"
        ]
    )
].copy()



all_scenarios = sorted(df_wide["SC"].unique())



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



# ----------------------- CREATION DU STYLE POUR LES FIGURES ---------------------
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

## Definition du style des types de traits
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





# ==========================================================================================
# FIGURE 1:
#       Comparaison 2 à 2 des scenarios naifs-non naifs
#       x: temps, en y le coefficient BC
#       couleur:  1 sets de parameters (scenario_0000, scenario_0001 etc)
#       type de trait: naif ou non naif

# Comment les différents comportements (native_naive, native_non_naive) 
# modifient-ils les dynamiques dans un système où le cerf est présent ?
# ==========================================================================================


pairs = [
    "C-M_BC",
    "C-P_BC",
    "M-P_BC"
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15,5),
    sharey=True
)

pair_titles = {
    "C-M_BC": "Caribou-Moose",
    "C-P_BC": "Caribou-Wolf",
    "M-P_BC": "Moose-Wolf"
}

colors = {
    "baseline": "black",
    "ii_native_naive": "tab:orange",
    "ii_native_non_naive": "tab:red"
}

cmpd_only = df_wide[
    df_wide["simulation"] == "CMPD"
]

for ax, pair in zip(axes, pairs):

    for exp_type in colors:

        tmp = (
            cmpd_only[
                cmpd_only["experiment_type"] == exp_type
            ]
            .groupby("time")[pair]
            .agg(["mean","std"])
            .reset_index()
        )

        ax.plot(
            tmp["time"],
            tmp["mean"],
            color=colors[exp_type],
            lw=2,
            label=exp_type
        )

        ax.fill_between(
            tmp["time"],
            tmp["mean"] - tmp["std"],
            tmp["mean"] + tmp["std"],
            color=colors[exp_type],
            alpha=0.15
        )

    ax.set_title(pair_titles[pair])

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

axes[0].set_ylabel(
    "Bhattacharyya coefficient"
)

axes[-1].legend()

plt.tight_layout()
plt.show()




# ==========================================================================================
# FIGURE 2:
# ii) b- Comparaison des chevauchements avec la ressource, risque de predation
# et ratio ressource/predation en fct du temps 
# chaque couleur = 1 sets de parameters (scenario_0000, scenario_0001 etc)
# et le type de trait = naif ou non naif

# Comment les différents comportements (native_naive, native_non_naive) 
# modifient-ils les dynamiques dans un système où le cerf est présent ?
# ==========================================================================================


cmpd = df_wide[
    df_wide["simulation"] == "CMPD"
].copy()


cmpd = cmpd.copy()

# Caribou
# cmpd["C_tradeoff_P"] = (
#     cmpd["C-Phi_C_ALIGN"]
#     - cmpd["C-riskP_ALIGN"]
# )

# cmpd["C_tradeoff_PM"] = (
#     cmpd["C-Phi_C_ALIGN"]
#     - cmpd["C-riskPM_ALIGN"]
# )

# cmpd["C_tradeoff_PMD"] = (
#     cmpd["C-Phi_C_ALIGN"]
#     - cmpd["C-riskPMD_ALIGN"]
# )



cmpd["C_tradeoff_P"] = (
    cmpd["C-IFDresource_ALIGN"]
    - cmpd["C-riskP_ALIGN"]
)

cmpd["C_tradeoff_PM"] = (
    cmpd["C-IFDresource_ALIGN"]
    - cmpd["C-riskPM_ALIGN"]
)

cmpd["C_tradeoff_PMD"] = (
    cmpd["C-IFDresource_ALIGN"]
    - cmpd["C-riskPMD_ALIGN"]
)




# Moose
# cmpd["M_tradeoff_P"] = (
#     cmpd["M-Phi_M_ALIGN"]
#     - cmpd["M-riskP_ALIGN"]
# )

# cmpd["M_tradeoff_PC"] = (
#     cmpd["M-Phi_M_ALIGN"]
#     - cmpd["M-riskPC_ALIGN"]
# )

# cmpd["M_tradeoff_PCD"] = (
#     cmpd["M-Phi_M_ALIGN"]
#     - cmpd["M-riskPCD_ALIGN"]
# )

cmpd["M_tradeoff_P"] = (
    cmpd["M-IFDresource_ALIGN"]
    - cmpd["M-riskP_ALIGN"]
)

cmpd["M_tradeoff_PC"] = (
    cmpd["M-IFDresource_ALIGN"]
    - cmpd["M-riskPC_ALIGN"]
)

cmpd["M_tradeoff_PCD"] = (
    cmpd["M-IFDresource_ALIGN"]
    - cmpd["M-riskPCD_ALIGN"]
)

colors = {
    "baseline": "black",
    "ii_native_naive": "tab:orange",
    "ii_native_non_naive": "tab:red"
}

fig, axes = plt.subplots(
    3,
    2,
    figsize=(14, 12),
    sharex=True,
    sharey="row"
)

# ============================================================
# Titres colonnes
# ============================================================

axes[0,0].set_title(
    "Caribou",
    fontsize=14,
    fontweight="bold"
)

axes[0,1].set_title(
    "Moose",
    fontsize=14,
    fontweight="bold"
)

# ============================================================
# Variables
# ============================================================

# resource_vars = [
#     "C-Phi_C_ALIGN",
#     "M-Phi_M_ALIGN"
# ]


resource_vars = [
    "C-IFDresource_ALIGN",
    "M-IFDresource_ALIGN"
]

risk_info = {
    0: {
        "P": ("C-riskP_ALIGN", "-"),
        "P+M": ("C-riskPM_ALIGN", "--"),
        "P+M+D": ("C-riskPMD_ALIGN", ":")
    },

    1: {
        "P": ("M-riskP_ALIGN", "-"),
        "P+C": ("M-riskPC_ALIGN", "--"),
        "P+C+D": ("M-riskPCD_ALIGN", ":")
    }
}

tradeoff_info = {
    0: {
        "P": ("C_tradeoff_P", "-"),
        "P+M": ("C_tradeoff_PM", "--"),
        "P+M+D": ("C_tradeoff_PMD", ":")
    },

    1: {
        "P": ("M_tradeoff_P", "-"),
        "P+C": ("M_tradeoff_PC", "--"),
        "P+C+D": ("M_tradeoff_PCD", ":")
    }
}


# ============================================================
# Ligne 1 : Alignement ressource
# ============================================================

for j, var in enumerate(resource_vars):

    ax = axes[0, j]

    for exp_type, color in colors.items():

        tmp = (
            cmpd[
                cmpd["experiment_type"] == exp_type
            ]
            .groupby("time")[var]
            .agg(["mean","std"])
            .reset_index()
        )

        ax.plot(
            tmp["time"],
            tmp["mean"],
            color=color,
            lw=2,
            label=exp_type
        )

        ax.fill_between(
            tmp["time"],
            tmp["mean"] - tmp["std"],
            tmp["mean"] + tmp["std"],
            color=color,
            alpha=0.15
        )

    ax.grid(alpha=0.3, linestyle=":")

# ============================================================
# Ligne 2 : Alignement prédateur
# ============================================================

risk_colors = {
    "baseline": "black",
    "ii_native_naive": "tab:orange",
    "ii_native_non_naive": "tab:red"
}

for j in [0, 1]:

    ax = axes[1, j]

    for label, (metric, ls) in risk_info[j].items():

        for exp_type, color in risk_colors.items():

            tmp = (
                cmpd[
                    cmpd["experiment_type"] == exp_type
                ]
                .groupby("time")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            ax.plot(
                tmp["time"],
                tmp["mean"],
                color=color,
                linestyle=ls,
                linewidth=2
            )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

    ax.axhline(
        0,
        color="grey",
        ls="--"
    )

# ============================================================
# Ligne 3 : Resource-Predation tradeoff
# ============================================================

for j in [0, 1]:

    ax = axes[2, j]

    for label, (metric, ls) in tradeoff_info[j].items():

        for exp_type, color in risk_colors.items():

            tmp = (
                cmpd[
                    cmpd["experiment_type"] == exp_type
                ]
                .groupby("time")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            ax.plot(
                tmp["time"],
                tmp["mean"],
                color=color,
                linestyle=ls,
                linewidth=2
            )

            ax.fill_between(
                tmp["time"],
                tmp["mean"] - tmp["std"],
                tmp["mean"] + tmp["std"],
                color=color,
                alpha=0.10
            )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )



# ============================================================
# Labels
# ============================================================
from matplotlib.lines import Line2D
axes[0,0].set_ylabel(
    "Resource alignment",
    fontsize=12
)

axes[1,0].set_ylabel(
    "Predator alignment",
    fontsize=12
)

axes[2,0].set_ylabel(
    r"Resource alignment $-$ Risk alignment",
    fontsize=12
)

for ax in axes[2,:]:
    ax.set_xlabel("Time")

# ============================================================
# Legende unique
# ============================================================

behavior_legend = [
    Line2D([0],[0], color="black", lw=2,
           label="baseline"),

    Line2D([0],[0], color="tab:orange", lw=2,
           label="naive"),

    Line2D([0],[0], color="tab:red", lw=2,
           label="non-naive")
]

risk_legend = [
    Line2D([0],[0], color="grey", lw=2,
           linestyle="-",
           label="P"),

    Line2D([0],[0], color="grey", lw=2,
           linestyle="--",
           label="P+M / P+C"),

    Line2D([0],[0], color="grey", lw=2,
           linestyle=":",
           label="P+M+D / P+C+D")
]

axes[0,1].legend(
    handles=behavior_legend,
    title="Behaviour",
    loc="upper right"
)

axes[1,1].legend(
    handles=risk_legend,
    title="Risk landscape",
    loc="upper right"
)

plt.tight_layout()
plt.show()





# ==========================================================================================
# FIGURE 3:




# ==========================================================================================

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CMPD seulement
# ============================================================

cmpd = df_wide[
    df_wide["simulation"] == "CMPD"
].copy()

# ============================================================
# Configuration
# ============================================================

behaviors = [
    "baseline",
    "ii_native_naive",
    "ii_native_non_naive"
]

behavior_titles = {
    "baseline": "Baseline",
    "ii_native_naive": "Native naive",
    "ii_native_non_naive": "Native non-naive"
}

species_info = [
    (
        "Caribou",
        "C-IFDresource_ALIGN",
        "C-riskPMD_ALIGN"
    ),
    (
        "Moose",
        "M-IFDresource_ALIGN",
        "M-riskPCD_ALIGN"
    ),
    (
        "Deer",
        "D-IFDresource_ALIGN",
        "D-riskPCM_ALIGN"
    )
    
]

# ============================================================
# Colormap temporelle
# ============================================================

norm = plt.Normalize(
    cmpd["time"].min(),
    cmpd["time"].max()
)

cmap = plt.cm.viridis

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15,10),
    sharex=False,
    sharey=False
)

last_scatter = None

# ============================================================
# Boucle
# ============================================================

for row, (
    species_name,
    resource_var,
    risk_var
) in enumerate(species_info):

    for col, behavior in enumerate(behaviors):

        ax = axes[row, col]

        sub = cmpd[
            cmpd["experiment_type"] == behavior
        ]

        for scen, tmp in sub.groupby("scenario"):

            tmp = tmp.sort_values("time")

            x = tmp[risk_var].values
            y = tmp[resource_var].values
            t = tmp["time"].values

            # ------------------------------------------------
            # Trajectoire colorée par le temps
            # ------------------------------------------------

            for k in range(len(tmp)-1):

                ax.plot(
                    x[k:k+2],
                    y[k:k+2],
                    color=cmap(norm(t[k])),
                    linewidth=1.8,
                    alpha=0.8
                )

            # ------------------------------------------------
            # Points temporels
            # ------------------------------------------------

            last_scatter = ax.scatter(
                x,
                y,
                c=t,
                cmap="viridis",
                norm=norm,
                s=30,
                edgecolors="none",
                alpha=0.9
            )

            # ------------------------------------------------
            # Point final
            # ------------------------------------------------

            ax.scatter(
                x[-1],
                y[-1],
                c=[t[-1]],
                cmap="viridis",
                norm=norm,
                s=90,
                edgecolors="black",
                linewidths=0.8,
                zorder=10
            )

            if "SC" in tmp.columns:

                ax.annotate(
                    tmp["SC"].iloc[0],
                    (x[-1], y[-1]),
                    fontsize=7,
                    xytext=(3,3),
                    textcoords="offset points"
                )

        # ------------------------------------------------
        # Habillage
        # ------------------------------------------------

        ax.grid(
            alpha=0.3,
            linestyle=":"
        )

        if row == 0:

            ax.set_title(
                behavior_titles[behavior],
                fontsize=13,
                fontweight="bold"
            )

        if col == 0:

            ax.set_ylabel(
                f"{species_name}\nResource alignment",
                fontsize=11
            )

        if row == 1:

            ax.set_xlabel(
                "Predation-risk alignment",
                fontsize=11
            )

# ============================================================
# Colorbar
# ============================================================

fig.subplots_adjust(
    right=0.88
)

cax = fig.add_axes(
    [0.90, 0.15, 0.02, 0.70]
)

cbar = fig.colorbar(
    last_scatter,
    cax=cax
)

cbar.set_label(
    "Time"
)


plt.show()




# ==========================================================================================
# FIGURE 4:

# ==========================================================================================


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CMPD seulement
# ============================================================

cmpd = df_wide[
    df_wide["simulation"] == "CMPD"
].copy()

# ============================================================
# Styles
# ============================================================

linestyles = {
    "baseline": "-",
    "ii_native_naive": "--",
    "ii_native_non_naive": ":"
}

behavior_order = [
    "baseline",
    "ii_native_naive",
    "ii_native_non_naive"
]

# ============================================================
# Espèces
# ============================================================

species_info = {
    "Caribou": {
        "resource": "C-IFDresource_ALIGN",
        "risk": "C-riskPMD_ALIGN"
    },

    "Moose": {
        "resource": "M-IFDresource_ALIGN",
        "risk": "M-riskPCD_ALIGN"
    },
    
    "Deer": {
        "resource": "D-IFDresource_ALIGN",
        "risk": "D-riskPCM_ALIGN"
        }
}

# ============================================================
# Scénarios
# ============================================================

scenario_order = sorted(
    cmpd["scenario"].unique()
)

# ============================================================
# Echelle de couleur
# ============================================================

norm = plt.Normalize(
    cmpd["time"].min(),
    cmpd["time"].max()
)

cmap = plt.cm.viridis

# ============================================================
# Figure 6x5
# ============================================================

fig, axes = plt.subplots(
    6,
    5,
    figsize=(18, 12),
    sharex=False,
    sharey=False
)

last_scatter = None

# ============================================================
# CARIBOU
# ============================================================

for idx, scen in enumerate(scenario_order):

    row = 0 if idx < 5 else 1
    col = idx % 5

    ax = axes[row, col]

    scen_data = cmpd[
        cmpd["scenario"] == scen
    ]

    for behavior in behavior_order:

        sub = scen_data[
            scen_data["experiment_type"] == behavior
        ]

        if len(sub) == 0:
            continue

        sub = sub.sort_values("time")

        x = sub["C-riskPMD_ALIGN"].values
        y = sub["C-IFDresource_ALIGN"].values
        t = sub["time"].values

        for k in range(len(sub)-1):

            ax.plot(
                x[k:k+2],
                y[k:k+2],
                color=cmap(norm(t[k])),
                linestyle=linestyles[behavior],
                lw=2
            )

        last_scatter = ax.scatter(
            x,
            y,
            c=t,
            cmap="viridis",
            norm=norm,
            s=25
        )

        ax.scatter(
            x[-1],
            y[-1],
            c=[t[-1]],
            cmap="viridis",
            norm=norm,
            s=70,
            edgecolors="black"
        )

    ax.set_title(
        scen.replace(
            "scenario_",
            "SC"
        ),
        fontsize=10
    )

    ax.grid(alpha=0.3, linestyle=":")

# ============================================================
# MOOSE
# ============================================================

for idx, scen in enumerate(scenario_order):

    row = 2 if idx < 5 else 3
    col = idx % 5

    ax = axes[row, col]

    scen_data = cmpd[
        cmpd["scenario"] == scen
    ]

    for behavior in behavior_order:

        sub = scen_data[
            scen_data["experiment_type"] == behavior
        ]

        if len(sub) == 0:
            continue

        sub = sub.sort_values("time")

        x = sub["M-riskPCD_ALIGN"].values
        y = sub["M-IFDresource_ALIGN"].values
        t = sub["time"].values

        for k in range(len(sub)-1):

            ax.plot(
                x[k:k+2],
                y[k:k+2],
                color=cmap(norm(t[k])),
                linestyle=linestyles[behavior],
                lw=2
            )

        last_scatter = ax.scatter(
            x,
            y,
            c=t,
            cmap="viridis",
            norm=norm,
            s=25
        )

        ax.scatter(
            x[-1],
            y[-1],
            c=[t[-1]],
            cmap="viridis",
            norm=norm,
            s=70,
            edgecolors="black"
        )

    ax.grid(alpha=0.3, linestyle=":")



# ============================================================
# DEER
# ============================================================

for idx, scen in enumerate(scenario_order):

    row = 4 if idx < 5 else 5
    col = idx % 5

    ax = axes[row, col]

    scen_data = cmpd[
        cmpd["scenario"] == scen
    ]

    for behavior in behavior_order:

        sub = scen_data[
            scen_data["experiment_type"] == behavior
        ]

        if len(sub) == 0:
            continue

        sub = sub.sort_values("time")

        x = sub["D-riskPCM_ALIGN"].values
        y = sub["D-IFDresource_ALIGN"].values
        t = sub["time"].values

        for k in range(len(sub)-1):

            ax.plot(
                x[k:k+2],
                y[k:k+2],
                color=cmap(norm(t[k])),
                linestyle=linestyles[behavior],
                lw=2
            )

        last_scatter = ax.scatter(
            x,
            y,
            c=t,
            cmap="viridis",
            norm=norm,
            s=25
        )

        ax.scatter(
            x[-1],
            y[-1],
            c=[t[-1]],
            cmap="viridis",
            norm=norm,
            s=70,
            edgecolors="black"
        )

    ax.grid(alpha=0.3, linestyle=":")

# ============================================================
# Labels de lignes
# ============================================================

axes[0,0].set_ylabel(
    "Caribou\nResource alignment"
)

axes[1,0].set_ylabel(
    "Caribou\nResource alignment"
)

axes[2,0].set_ylabel(
    "Moose\nResource alignment"
)

axes[3,0].set_ylabel(
    "Moose\nResource alignment"
)

axes[4,0].set_ylabel(
    "Deer\nResource alignment"
)

axes[5,0].set_ylabel(
    "Deer\nResource alignment"
)

# ============================================================
# Labels X
# ============================================================

for ax in axes[5,:]:
    ax.set_xlabel(
        "Predation-risk alignment"
    )

# ============================================================
# Legende comportements
# ============================================================

legend_elements = [

    Line2D(
        [0],[0],
        color="black",
        linestyle="-",
        lw=2,
        label="Baseline"
    ),

    Line2D(
        [0],[0],
        color="black",
        linestyle="--",
        lw=2,
        label="Native naive"
    ),

    Line2D(
        [0],[0],
        color="black",
        linestyle=":",
        lw=2,
        label="Native non-naive"
    )
]

axes[0,4].legend(
    handles=legend_elements,
    frameon=False,
    title="Behaviour"
)

# ============================================================
# Colorbar
# ============================================================

fig.subplots_adjust(
    right=0.90
)

cax = fig.add_axes(
    [0.92, 0.15, 0.015, 0.70]
)

cbar = fig.colorbar(
    last_scatter,
    cax=cax
)

cbar.set_label(
    "Time"
)


plt.show()



# ==========================================================================================
# FIGURE 6:
    # Caribou and moose pour chaque scenario, resource alignement et predaiton-risk alignement
# ==========================================================================================
 

alt = cmpd[
    cmpd["experiment_type"] != "baseline"
]

scenarios = sorted(
    alt["scenario"].unique()
)

times = np.sort(
    alt["time"].unique()
)

selected_times = np.unique(
    times[
        np.linspace(
            0,
            len(times)-1,
            5,
            dtype=int
        )
    ]
)

sizes = np.linspace(
    30,
    120,
    len(selected_times)
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18,8),
    sharex=True,
    sharey=True
)

axes = axes.ravel()

species_info = {

    "Caribou": {
        "resource": "C-IFDresource_ALIGN",
        "risk": "C-riskPMD_ALIGN",
        "color": "tab:blue"
    },

    "Moose": {
        "resource": "M-IFDresource_ALIGN",
        "risk": "M-riskPCD_ALIGN",
        "color": "tab:orange"
    }
}

linestyles = {
    "ii_native_naive": "--",
    "ii_native_non_naive": ":"
}

for ax, scen in zip(axes, scenarios):

    scen_data = alt[
        alt["scenario"] == scen
    ]

    for species, info in species_info.items():

        for behavior in [
            "ii_native_naive",
            "ii_native_non_naive"
        ]:

            sub = scen_data[
                scen_data["experiment_type"]
                == behavior
            ]

            if len(sub) == 0:
                continue

            sub = (
                sub.sort_values("time")
                .set_index("time")
                .loc[selected_times]
                .reset_index()
            )

            ax.plot(
                sub[info["risk"]],
                sub[info["resource"]],
                color=info["color"],
                linestyle=linestyles[behavior],
                lw=2
            )

            for k in range(len(sub)):

                ax.scatter(
                    sub[info["risk"]].iloc[k],
                    sub[info["resource"]].iloc[k],
                    s=sizes[k],
                    color=info["color"],
                    alpha=0.85
                )

    ax.set_title(
        scen.replace(
            "scenario_",
            "SC"
        )
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

for ax in axes[5:]:
    ax.set_xlabel(
        "Predation-risk alignment"
    )

for ax in axes[::5]:
    ax.set_ylabel(
        "Resource alignment"
    )

legend_species = [
    Line2D(
        [0],[0],
        color="tab:blue",
        lw=3,
        label="Caribou"
    ),
    Line2D(
        [0],[0],
        color="tab:orange",
        lw=3,
        label="Moose"
    )
]

legend_behaviour = [
    Line2D(
        [0],[0],
        color="black",
        linestyle="--",
        lw=2,
        label="Native naive"
    ),
    Line2D(
        [0],[0],
        color="black",
        linestyle=":",
        lw=2,
        label="Native non-naive"
    )
]

# ============================================================
# Légende temps (taille des points)
# ============================================================

time_legend = [

    Line2D(
        [0], [0],
        marker="o",
        color="grey",
        linestyle="",
        markerfacecolor="grey",
        markersize=np.sqrt(size),
        label=label
    )

    for size, label in zip(
        sizes,
        ["t0", "25%", "50%", "75%", "100%"]
    )
]

axes[-1].legend(
    handles=(
        legend_species
        + legend_behaviour
        + time_legend
    ),
    frameon=False,
    fontsize=8,
    title="Species / Behaviour / Time"
)

plt.tight_layout()
plt.show()



# ==========================================================================================
# FIGURE 7:
# ==========================================================================================
alt = cmpd[
    cmpd["experiment_type"] != "baseline"
]

scenarios = sorted(
    alt["scenario"].unique()
)

times = np.sort(
    alt["time"].unique()
)

selected_times = np.unique(
    times[
        np.linspace(
            0,
            len(times)-1,
            5,
            dtype=int
        )
    ]
)

sizes = np.linspace(
    30,
    120,
    len(selected_times)
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18,8),
    sharex=True,
    sharey=True
)

axes = axes.ravel()

species_info = {

    "Caribou": {
        "resource": "C-IFDresource_ALIGN",
        "risk": "C-riskPMD_ALIGN",
        "color": "tab:blue"
    },

    "Moose": {
        "resource": "M-IFDresource_ALIGN",
        "risk": "M-riskPCD_ALIGN",
        "color": "tab:orange"
    },

    "Deer": {
        "resource": "D-IFDresource_ALIGN",
        "risk": "D-riskPCM_ALIGN",
        "color": "tab:green"
    }
}


linestyles = {
    "ii_native_naive": "--",
    "ii_native_non_naive": ":"
}

for ax, scen in zip(axes, scenarios):

    scen_data = alt[
        alt["scenario"] == scen
    ]

    for species, info in species_info.items():

        for behavior in [
            "ii_native_naive",
            "ii_native_non_naive"
        ]:

            sub = scen_data[
                scen_data["experiment_type"]
                == behavior
            ]

            if len(sub) == 0:
                continue

            sub = (
                sub.sort_values("time")
                .set_index("time")
                .loc[selected_times]
                .reset_index()
            )

            ax.plot(
                sub[info["risk"]],
                sub[info["resource"]],
                color=info["color"],
                linestyle=linestyles[behavior],
                lw=2
            )

            for k in range(len(sub)):

                ax.scatter(
                    sub[info["risk"]].iloc[k],
                    sub[info["resource"]].iloc[k],
                    s=sizes[k],
                    color=info["color"],
                    alpha=0.85
                )

    ax.set_title(
        scen.replace(
            "scenario_",
            "SC"
        )
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

for ax in axes[5:]:
    ax.set_xlabel(
        "Predation-risk alignment"
    )

for ax in axes[::5]:
    ax.set_ylabel(
        "Resource alignment"
    )

legend_species = [

    Line2D(
        [0],[0],
        color="tab:blue",
        lw=3,
        label="Caribou"
    ),

    Line2D(
        [0],[0],
        color="tab:orange",
        lw=3,
        label="Moose"
    ),

    Line2D(
        [0],[0],
        color="tab:green",
        lw=3,
        label="Deer"
    )
]

legend_behaviour = [
    Line2D(
        [0],[0],
        color="black",
        linestyle="--",
        lw=2,
        label="Native naive"
    ),
    Line2D(
        [0],[0],
        color="black",
        linestyle=":",
        lw=2,
        label="Native non-naive"
    )
]

# ============================================================
# Légende temps (taille des points)
# ============================================================

time_legend = [

    Line2D(
        [0], [0],
        marker="o",
        color="grey",
        linestyle="",
        markerfacecolor="grey",
        markersize=np.sqrt(size),
        label=label
    )

    for size, label in zip(
        sizes,
        ["t0", "25%", "50%", "75%", "100%"]
    )
]

axes[-1].legend(
    handles=(
        legend_species
        + legend_behaviour
        + time_legend
    ),
    frameon=False,
    fontsize=8,
    title="Species / Behaviour / Time"
)

plt.tight_layout()
plt.show()



# ==========================================================================================
# FIGURE 8:
    # Comparaison naive vs non-naive au temps final
    # Couleur = scénario (SC0, SC1, ...)
    # Forme = comportement
# ==========================================================================================

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

tmax = df_wide["time"].max()

df_final = df_wide[
    (df_wide["time"] == tmax)
    &
    (
        df_wide["experiment_type"].isin([
            "ii_native_naive",
            "ii_native_non_naive"
        ])
    )
].copy()

pairs = [
    "C-M_BC",
    "C-P_BC",
    "M-P_BC"
]

pair_titles = {
    "C-M_BC": "Caribou-Moose",
    "C-P_BC": "Caribou-Wolf",
    "M-P_BC": "Moose-Wolf"
}

# ============================================================
# Couleurs des scénarios
# ============================================================

scenarios = sorted(
    df_final["SC"].unique()
)


# ============================================================
# Marqueurs
# ============================================================

markers = {
    "ii_native_naive": "o",
    "ii_native_non_naive": "^"
}

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(15,5),
    sharey=False
)

for ax, pair in zip(axes, pairs):

    for exp_type in [
        "ii_native_naive",
        "ii_native_non_naive"
    ]:

        sub = df_final[
            df_final["experiment_type"] == exp_type
        ]

        for _, row in sub.iterrows():

            ax.scatter(
                row["log_ratio"],
                row[pair],
                color=scenario_colors[row["SC"]],
                marker=markers[exp_type],
                s=120,
                edgecolors="black",
                linewidth=0.7,
                alpha=0.9
            )

    ax.set_title(
        pair_titles[pair],
        fontsize=13,
        fontweight="bold"
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
# Légende scénarios
# ============================================================

scenario_legend = [

    Line2D(
        [0],[0],
        marker="o",
        color=scenario_colors[sc],
        linestyle="",
        markersize=10,
        label=sc
    )

    for sc in scenarios
]


legend_scenarios = axes[-1].legend(
    handles=scenario_legend,
    title="Scenario",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    frameon=False
)

# ============================================================
# Légende comportements
# ============================================================

behaviour_legend = [

    Line2D(
        [0],[0],
        marker="o",
        color="black",
        linestyle="",
        markersize=10,
        label="Native naive"
    ),

    Line2D(
        [0],[0],
        marker="^",
        color="black",
        linestyle="",
        markersize=10,
        label="Native non-naive"
    )
]

legend_behaviour = axes[-1].legend(
    handles=behaviour_legend,
    title="Behaviour",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.45),
    frameon=False
)

axes[-1].add_artist(legend_scenarios)

plt.tight_layout(
    rect=[0, 0, 0.85, 1]
)
plt.show()







# ==========================================================================================
# FIGURE 9:
    # Comparaison naive vs non-naive au temps final
    # Couleur = scénario (SC0, SC1, ...)
    # Forme = comportement
    ## Figure en 2 lignes et 3 colonnes
# ==========================================================================================


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

tmax = df_wide["time"].max()

df_final = df_wide[
    (df_wide["time"] == tmax)
    &
    (
        df_wide["experiment_type"].isin([
            "ii_native_naive",
            "ii_native_non_naive"
        ])
    )
].copy()




pairs = [
    p for p in [
        "C-M_BC",
        "C-P_BC",
        "C-D_BC",
        "M-P_BC",
        "M-D_BC",
        "P-D_BC"
    ]
    if p in df_wide.columns
]

## On donne les noms complets pour les figures
pair_titles = {
    "C-M_BC": "Caribou-Moose",
    "C-P_BC": "Caribou-Wolf",
    "C-D_BC": "Caribou-Deer",
    "M-P_BC": "Moose-Wolf",
    "M-D_BC": "Moose-Deer",
    "P-D_BC": "Wolf-Deer"
}

# ============================================================
# Couleurs des scénarios
# ============================================================

scenarios = sorted(
    df_final["SC"].unique()
)


# ============================================================
# Marqueurs
# ============================================================

markers = {
    "ii_native_naive": "o",
    "ii_native_non_naive": "^"
}

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12,8),
    sharey=False
)

axes = axes.ravel()

for ax, pair in zip(axes, pairs):

    for exp_type in [
        "ii_native_naive",
        "ii_native_non_naive"
    ]:

        sub = df_final[
            df_final["experiment_type"] == exp_type
        ]

        for _, row in sub.iterrows():

            ax.scatter(
                row["log_ratio"],
                row[pair],
                color=scenario_colors[row["SC"]],
                marker=markers[exp_type],
                s=120,
                edgecolors="black",
                linewidth=0.7,
                alpha=0.9
            )


    ax.set_title(
        pair_titles[pair],
        fontsize=13,
        fontweight="bold"
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

axes[3].set_ylabel(
    "Bhattacharyya coefficient"
)

# ============================================================
# Légende scénarios
# ============================================================

scenario_legend = [

    Line2D(
        [0],[0],
        marker="o",
        color=scenario_colors[sc],
        linestyle="",
        markersize=10,
        label=sc
    )

    for sc in sorted(scenarios)
]

# ============================================================
# Légende comportements
# ============================================================

behaviour_legend = [

    Line2D(
        [0],[0],
        marker="o",
        color="black",
        linestyle="",
        markersize=10,
        label="Native naive"
    ),

    Line2D(
        [0],[0],
        marker="^",
        color="black",
        linestyle="",
        markersize=10,
        label="Native non-naive"
    )
]

legend1 = axes[-1].legend(
    handles=scenario_legend,
    title="Scenario",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    frameon=False
)


legend2 = axes[-1].legend(
    handles=behaviour_legend,
    title="Behaviour",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.45),
    frameon=False
)

axes[-1].add_artist(legend1)

plt.tight_layout(
    rect=[0, 0, 0.82, 1]
)

plt.show()





# ==========================================================================================
# FIGURE 10:
# ==========================================================================================


scenarios = sorted(
    df_wide["SC"].dropna().unique()
)


pairs = [
    p for p in [
        "C-M_BC",
        "C-P_BC",
        "C-D_BC",
        "M-P_BC",
        "M-D_BC",
        "P-D_BC"
    ]
    if p in df_wide.columns
]

## On doonne les noms complets pour les figures
col_titles = [
    "Caribou-Moose",
    "Caribou-Wolf",
    "Caribou-Deer",
    "Moose-Wolf",
    "Moose-Deer", 
    "Wolf-Deer"
]

t_zoom = 60

fig, axes = plt.subplots(
    2,
    len(pairs),
    figsize=(25,10),
    sharey="row"
)

for j, (pair, title) in enumerate(
    zip(pairs, col_titles)
):

    ax_full = axes[0, j]
    ax_zoom = axes[1, j]

    for (sim, sc, exp_type), sub in (
        df_wide.groupby(
            ["simulation",
             "SC",
             "experiment_type"]
        )
    ):

        sub = sub.sort_values("time")

        color = scenario_colors[sc]

        ls = experiment_styles[
            exp_type
        ]["linestyle"]

        lw = experiment_styles[
            exp_type
        ]["lw"]

        alpha = experiment_styles[
            exp_type
        ]["alpha"]

        # ----------------------------------------------------
        # Série complète
        # ----------------------------------------------------

        ax_full.plot(
            sub["time"],
            sub[pair],
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha
        )

        # ----------------------------------------------------
        # Zoom
        # ----------------------------------------------------

        sub_zoom = sub[
            sub["time"] >= t_zoom
        ]

        ax_zoom.plot(
            sub_zoom["time"],
            sub_zoom[pair],
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha
        )

    # --------------------------------------------------------
    # Habillage
    # --------------------------------------------------------

    ax_full.set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    ax_full.grid(
        alpha=0.3,
        linestyle=":"
    )

    ax_zoom.grid(
        alpha=0.3,
        linestyle=":"
    )

    ax_zoom.set_xlabel(
        "Time"
    )

# ============================================================
# Labels
# ============================================================

axes[0,0].set_ylabel(
    "BC overlap\n(full simulation)"
)

axes[1,0].set_ylabel(
    f"BC overlap\n(t ≥ {t_zoom})"
)

# ============================================================
# Légende comportement
# ============================================================

legend_items = [

    Line2D(
        [0],[0],
        color="black",
        linestyle="-",
        lw=3,
        label="baseline"
    ),

    Line2D(
        [0],[0],
        color="black",
        linestyle="--",
        lw=2,
        label="ii_native_naive"
    ),

    Line2D(
        [0],[0],
        color="black",
        linestyle=":",
        lw=2,
        label="ii_native_non_naive"
    )
]

# ==============
# Légende de scenarios
# ==============

scenario_legend = [
    Line2D(
        [0], [0],
        color=scenario_colors[sc],
        lw=3,
        label=sc
    )
    for sc in scenarios
]

fig.legend(
    handles=scenario_legend,
    title="Scenario",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.88, 0.90)
)

fig.legend(
    handles=legend_items,
    title="Behaviour",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.88, 0.45)
)


plt.tight_layout(
    rect=[0, 0, 0.85, 0.95]
)
plt.show()






# ==========================================================================================
# FIGURE 10:
# Delta BC entre comportements
# non-naive - naive
# ==========================================================================================

pairs = [
    p for p in [
        "C-M_BC",
        "C-D_BC",
        "M-D_BC",
        "C-P_BC",
        "M-P_BC",
        "P-D_BC"
    ]
    if p in df_wide.columns
]

## On doonne les noms complets pour les figures
col_titles = [
    "Caribou-Moose",
    "Caribou-Deer",
    "Moose-Deer",
    "Caribou-Wolf",
    "Moose-Wolf",
    "Wolf-Deer"
]

# ------------------------------------------------------------
# Séparer naïf / non-naïf
# ------------------------------------------------------------

naive = df_wide[
    df_wide["experiment_type"]
    == "ii_native_naive"
].copy()

non_naive = df_wide[
    df_wide["experiment_type"]
    == "ii_native_non_naive"
].copy()

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

merge_cols = [
    "simulation",
    "scenario",
    "SC",
    "time"
]

df_delta = naive.merge(
    non_naive,
    on=merge_cols,
    suffixes=("_naive", "_non_naive"),
    validate="one_to_one"
)


# ------------------------------------------------------------
# Calcul ΔBC
# ------------------------------------------------------------

for pair in pairs:

    df_delta[f"delta_{pair}"] = (
        df_delta[f"{pair}_non_naive"]
        -
        df_delta[f"{pair}_naive"]
    )

# ------------------------------------------------------------
# Couleurs scénarios
# ------------------------------------------------------------

scenarios = sorted(
    df_delta["SC"].unique()
)


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    #len(pairs),
    figsize=(18,8),
    sharex=True
    #sharey=True
)
axes = axes.ravel()

for ax, pair, title in zip(
    axes,
    pairs,
    col_titles
):

    delta_col = f"delta_{pair}"

    for sc, sub in df_delta.groupby("SC"):

        sub = sub.sort_values("time")

        ax.plot(
            sub["time"],
            sub[delta_col],
            color=scenario_colors[sc],
            linewidth=2,
            alpha=0.9,
            label=sc
        )


    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

axes[0].set_ylabel(
    r"$\Delta BC =$ non-naive $-$ naive"
)

axes[3].set_ylabel(
    r"$\Delta BC =$ non-naive $-$ naive"
)

for ax in axes[3:]:
    ax.set_xlabel("Time")

# ------------------------------------------------------------
# Légende scénarios
# ------------------------------------------------------------

legend_items = [

    Line2D(
        [0],[0],
        color=scenario_colors[sc],
        lw=3,
        label=sc
    )

    for sc in scenarios
]

fig.legend(
    handles=legend_items,
    title="Scenario",
    loc="center right",
    frameon=False
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()




# ==========================================================================================
# FIGURE 11:
# 
# 
# ==========================================================================================

## --- POur les 3 especes de proie
## 1ere ligne: alignement ressource
## 2nde ligne: alignement predateur
## 3eme ligne: ratio des 2


species = ["Caribou", "Moose", "Deer"]

# resource_align = {
#     "Caribou": "C-Phi_C_ALIGN",
#     "Moose":   "M-Phi_M_ALIGN",
#     "Deer":    "D-Phi_D_ALIGN",
# }

resource_align = {
    "Caribou": "C-IFDresource_ALIGN",
    "Moose":   "M-IFDresource_ALIGN",
    "Deer":    "D-IFDresource_ALIGN",
}


predator_align = {
    "Caribou": {
        "P"      : ("C-riskP_ALIGN", "-"),
        "P+M"    : ("C-riskPM_ALIGN", "--"),
        "P+M+D"  : ("C-riskPMD_ALIGN", ":")
    },

    "Moose": {
        "P"      : ("M-riskP_ALIGN", "-"),
        "P+C"    : ("M-riskPC_ALIGN", "--"),
        "P+C+D"  : ("M-riskPCD_ALIGN", ":")
    },

    "Deer": {
        "P"      : ("D-riskP_ALIGN", "-"),
        "P+C"    : ("D-riskPC_ALIGN", "--"),
        "P+C+M"  : ("D-riskPCM_ALIGN", ":")
    }
}


# # ----------- 1 espece a la fois, x 10 scenarios

scenario_list = [
    sc
    for sc in sorted(df_wide["SC"].unique())
    if sc != "BASELINE"
]

species = "Caribou"

res_var = resource_align[species]


# ----- TEST
species = "Caribou"

res_var = resource_align[species]


scenario_list = sorted(df_wide["SC"].unique())

fig, axes = plt.subplots(
    3,
    len(scenario_list),
    figsize=(3 * len(scenario_list), 10),
    sharex=True,
    sharey="row"
)

for col, sc in enumerate(scenario_list):

    sub_sc = df_wide[
        (df_wide["SC"] == sc) &
        (df_wide["simulation"] == "CMPD")
    ]

    for (sim, exp_type), sub in (
        sub_sc.groupby(
            ["simulation", "experiment_type"]
        )
    ):

        sub = sub.sort_values("time")


        style = experiment_styles[exp_type]
        color = style["color"]

        axes[0, col].plot(
            sub["time"],
            sub[res_var],
            color=style["color"],
            lw=style["lw"],
            alpha=style["alpha"],
            linestyle=style["linestyle"]
        )



        #
        # ligne 2 et 3
        #
        for pred_name, (pred_col, pred_ls) in (
            predator_align[species].items()
        ):

            if pred_col not in sub.columns:
                continue

            #
            # alignement prédateur
            #
            axes[1, col].plot(
                sub["time"],
                sub[pred_col],
                color=color,
                linestyle=pred_ls,
                lw=style["lw"],
                alpha=style["alpha"]
            )

            #
            # delta
            #
            delta = sub[res_var] - sub[pred_col]

            axes[2, col].plot(
                sub["time"],
                delta,
                color=color,
                linestyle=pred_ls,
                lw=style["lw"],
                alpha=style["alpha"]
            )

    axes[0, col].set_title(sc)


#
# labels des lignes
#
axes[0,0].set_ylabel("Resource\nalignment")
axes[1,0].set_ylabel("Predator\nalignment")
axes[2,0].set_ylabel("Resource alignment\n− Predator alignment")


#
# axe x
#
for ax in axes[2,:]:
    ax.set_xlabel("Time")

#
# ligne de référence
#
for ax in axes[2,:]:
    ax.axhline(
        0,
        color="black",
        ls="--",
        lw=1
    )

#
# grille
#
for ax in axes.flat:
    ax.grid(
        alpha=0.25,
        linestyle=":"
    )

from matplotlib.lines import Line2D

behaviour_handles = [
    Line2D(
        [0], [0],
        color=experiment_styles[b]["color"],
        lw=3,
        label=b
    )
    for b in experiment_styles
]

risk_handles = [
    Line2D(
        [0],[0],
        color="black",
        lw=2,
        ls=ls,
        label=name
    )
    for name, (_, ls) in predator_align[species].items()
]


leg1 = axes[0,-1].legend(
    handles=behaviour_handles,
    title="Behaviour",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)

axes[0,-1].add_artist(leg1)

axes[1,-1].legend(
    handles=risk_handles,
    title="Predator field",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)


fig.suptitle(
    f"{species}",
    fontsize=18
)

plt.tight_layout(
    rect=[0.05, 0, 0.92, 0.97]
)

plt.show()



# ==========================================================================================
# FIGURE 12:
# 
# 
# ==========================================================================================


species = "Moose"

res_var = resource_align[species]

scenario_list = sorted(df_wide["SC"].unique())

fig, axes = plt.subplots(
    3,
    len(scenario_list),
    figsize=(3 * len(scenario_list), 10),
    sharex=True,
    sharey="row"
)

for col, sc in enumerate(scenario_list):

    sub_sc = df_wide[
    (df_wide["SC"] == sc) &
    (df_wide["simulation"] == "CMPD")
]

    for (sim, exp_type), sub in (
        sub_sc.groupby(
            ["simulation", "experiment_type"]
        )
    ):

        sub = sub.sort_values("time")

        style = experiment_styles[exp_type]
        color = style["color"]

        axes[0, col].plot(
            sub["time"],
            sub[res_var],
            color=style["color"],
            lw=style["lw"],
            alpha=style["alpha"],
            linestyle=style["linestyle"]
        )

        #
        # ligne 2 et 3
        #
        for pred_name, (pred_col, pred_ls) in (
            predator_align[species].items()
        ):

            if pred_col not in sub.columns:
                continue

            #
            # alignement prédateur
            #
            axes[1, col].plot(
                sub["time"],
                sub[pred_col],
                color=color,
                linestyle=pred_ls,
                lw=style["lw"],
                alpha=style["alpha"]
            )

            #
            # delta
            #
            delta = sub[res_var] - sub[pred_col]

            axes[2, col].plot(
                sub["time"],
                delta,
                color=color,
                linestyle=pred_ls,
                lw=style["lw"],
                alpha=style["alpha"]
            )

    axes[0, col].set_title(sc)


#
# labels des lignes
#
axes[0,0].set_ylabel("Resource\nalignment")
axes[1,0].set_ylabel("Predator\nalignment")
axes[2,0].set_ylabel("Resource alignment\n− Predator alignment")

#
# axe x
#
for ax in axes[2,:]:
    ax.set_xlabel("Time")

#
# ligne de référence
#
for ax in axes[2,:]:
    ax.axhline(
        0,
        color="black",
        ls="--",
        lw=1
    )

#
# grille
#
for ax in axes.flat:
    ax.grid(
        alpha=0.25,
        linestyle=":"
    )

from matplotlib.lines import Line2D

behaviour_handles = [
    Line2D(
        [0], [0],
        color=experiment_styles[b]["color"],
        lw=3,
        label=b
    )
    for b in experiment_styles
]

risk_handles = [
    Line2D(
        [0],[0],
        color="black",
        lw=2,
        ls=ls,
        label=name
    )
    for name, (_, ls) in predator_align[species].items()
]


leg1 = axes[0,-1].legend(
    handles=behaviour_handles,
    title="Behaviour",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)

axes[0,-1].add_artist(leg1)

axes[1,-1].legend(
    handles=risk_handles,
    title="Predator field",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)


fig.suptitle(
    f"{species}",
    fontsize=18
)

plt.tight_layout(
    rect=[0.05, 0, 0.92, 0.97]
)

plt.show()



# ==========================================================================================
# FIGURE 13:
# 
# 
# ==========================================================================================

## ---------- Deer

species = "Deer"

res_var = resource_align[species]


scenario_list = sorted(df_wide["SC"].unique())

fig, axes = plt.subplots(
    3,
    len(scenario_list),
    figsize=(3 * len(scenario_list), 10),
    sharex=True,
    sharey="row"
)

for col, sc in enumerate(scenario_list):

    sub_sc = df_wide[
        (df_wide["SC"] == sc) &
        (df_wide["simulation"] == "CMPD")
    ]

    for (sim, exp_type), sub in (
        sub_sc.groupby(
            ["simulation", "experiment_type"]
        )
    ):

        style = experiment_styles[exp_type]
        color = style["color"]

        axes[0, col].plot(
            sub["time"],
            sub[res_var],
            color=style["color"],
            lw=style["lw"],
            alpha=style["alpha"],
            linestyle=style["linestyle"]
        )
        

        #
        # ligne 2 et 3
        #
        for pred_name, (pred_col, pred_ls) in (
            predator_align[species].items()
        ):

            if pred_col not in sub.columns:
                continue

            #
            # alignement prédateur
            #
            axes[1, col].plot(
                sub["time"],
                sub[pred_col],
                color=color,
                linestyle=pred_ls,
                lw=style["lw"],
                alpha=style["alpha"]
            )

            #
            # delta
            #
            delta = sub[res_var] - sub[pred_col]

            axes[2, col].plot(
                sub["time"],
                delta,
                color=color,
                linestyle=pred_ls,
                lw=style["lw"],
                alpha=style["alpha"]
            )

    axes[0, col].set_title(sc)


#
# labels des lignes
#
axes[0,0].set_ylabel("Resource\nalignment")
axes[1,0].set_ylabel("Predator\nalignment")
axes[2,0].set_ylabel("Resource alignment\n− Predator alignment")

#
# axe x
#
for ax in axes[2,:]:
    ax.set_xlabel("Time")

#
# ligne de référence
#
for ax in axes[2,:]:
    ax.axhline(
        0,
        color="black",
        ls="--",
        lw=1
    )

#
# grille
#
for ax in axes.flat:
    ax.grid(
        alpha=0.25,
        linestyle=":"
    )

from matplotlib.lines import Line2D

behaviour_handles = [
    Line2D(
        [0], [0],
        color=experiment_styles[b]["color"],
        lw=3,
        label=b
    )
    for b in experiment_styles
]

risk_handles = [
    Line2D(
        [0],[0],
        color="black",
        lw=2,
        ls=ls,
        label=name
    )
    for name, (_, ls) in predator_align[species].items()
]


leg1 = axes[0,-1].legend(
    handles=behaviour_handles,
    title="Behaviour",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)

axes[0,-1].add_artist(leg1)

axes[1,-1].legend(
    handles=risk_handles,
    title="Predator field",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)


fig.suptitle(
    f"{species}",
    fontsize=18
)

plt.tight_layout(
    rect=[0.05, 0, 0.92, 0.97]
)

plt.show()



# ==========================================================================================
# FIGURE 14:
# 
# 
# ==========================================================================================

for sc in sorted(df_wide["SC"].unique()):
    print("\n", sc)
    print(
        sorted(
            df_wide.loc[
                df_wide["SC"] == sc,
                "experiment_type"
            ].unique()
        )
    )

(
    df_wide[
        ["SC", "scenario", "simulation", "experiment_type"]
    ]
    .drop_duplicates()
    .sort_values(["SC", "scenario"])
)

(
    df_wide[
        ["scenario", "SC"]
    ]
    .drop_duplicates()
    .sort_values("SC")
)




print(
    df_wide.loc[
        df_wide["experiment_type"] == "baseline",
        ["scenario", "SC"]
    ]
    .drop_duplicates()
)

print(
    sorted(
        df_wide["SC"].unique()
    )
)




# # ------------- Adaptaiton de cette figure pour comparer CMPD naif et non naif
# variables = [
#     "C-riskP_ALIGN",
#     "C-riskPM_ALIGN",
#     "C-riskPMD_ALIGN"
# ]

# titles = [
#     "Risk P",
#     "Risk P+M",
#     "Risk P+M+D"
# ]

# cmpd = df_wide[
#     df_wide["simulation"] == "CMPD"
# ]

# cmpd = cmpd[
#     cmpd["experiment_type"].isin([
#         "ii_native_naive",
#         "ii_native_non_naive"
#     ])
# ]

# fig, axes = plt.subplots(
#     1,
#     3,
#     figsize=(15,5),
#     sharey=True
# )

# for ax, var, title in zip(axes, variables, titles):

#     for (sc, exp_type), sub in cmpd.groupby(
#         ["SC", "experiment_type"]
#     ):

#         sub = sub.sort_values("time")

#         ls = (
#             "--"
#             if exp_type == "ii_native_naive"
#             else "-"
#         )

#         ax.plot(
#             sub["time"],
#             sub[var],
#             color=scenario_colors[sc],
#             linestyle=ls,
#             linewidth=2,
#             alpha=0.9
#         )


#     ax.set_title(title)
#     ax.set_xlabel("Time")
#     ax.grid(alpha=0.3, linestyle=":")

# axes[0].set_ylabel(
#     "Predation-risk alignment"
# )

# # ---------- légende comportement

# behaviour_legend = [

#     Line2D(
#         [0], [0],
#         color="black",
#         linestyle="--",
#         lw=2,
#         label="Native naive"
#     ),

#     Line2D(
#         [0], [0],
#         color="black",
#         linestyle="-",
#         lw=2,
#         label="Native non-naive"
#     )
# ]

# scenario_handles = [

#     Line2D(
#         [0], [0],
#         color=scenario_colors[sc],
#         lw=3,
#         label=sc
#     )

#     for sc in sorted(cmpd["SC"].unique())
# ]

# fig.legend(
#     handles=scenario_handles,
#     title="Scenario",
#     bbox_to_anchor=(1.02, 0.5),
#     loc="center left"
# )

# plt.tight_layout(rect=[0,0,0.88,1])
# plt.show()




# ==========================================================================================
# FIGURE 15: Ratio ressource / prédateur au temps final
# 
# 
# ==========================================================================================

tmax = df_wide["time"].max()

df_final = (
    df_wide[
        df_wide["time"] == tmax
    ]
)

ratios = [
    "C_IFD_ratio_P",
    "M_IFD_ratio_P",
    "D_IFD_ratio_P"
]

scenario_order = (
    df_final[
        ["SC","c_ratio"]
    ]
    .drop_duplicates()
    .sort_values("c_ratio")
)


titles = [
    "Caribou",
    "Moose",
    "Deer"
]

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# --------------------------------------------------
# Données finales
# --------------------------------------------------

tmax = df_wide["time"].max()

df_final = df_wide[
    (df_wide["time"] == tmax)
    &
    (df_wide["simulation"] == "CMPD")
    &
    (
        df_wide["experiment_type"].isin([
            "ii_native_naive",
            "ii_native_non_naive"
        ])
    )
].copy()


baseline = df_wide[
    (df_wide["time"] == tmax)
    &
    (df_wide["simulation"] == "CMPD")
    &
    (df_wide["experiment_type"] == "baseline")
].copy()

# --------------------------------------------------
# Variables
# --------------------------------------------------
species_info = [
    {
        "title": "Caribou",
        "x": "c_ratio",
        "y": "C_IFD_ratio_P"
    },
    {
        "title": "Moose",
        "x": "m_ratio",
        "y": "M_IFD_ratio_P"
    },
    {
        "title": "Deer",
        "x": "d_ratio",
        "y": "D_IFD_ratio_P"
    }
]
# --------------------------------------------------
# Couleurs des scénarios
# --------------------------------------------------

scenario_order = (
    df_final[
        ["SC", "c_ratio"]
    ]
    .drop_duplicates()
    .sort_values("c_ratio")
)

scenarios = scenario_order["SC"].unique()


## trier les scenarios selon ordre croissant 
scenarios = sorted(
    df_final["SC"].unique(),
    key=lambda x: int(x.replace("SC", ""))
)




# --------------------------------------------------
# Marqueurs
# --------------------------------------------------

markers = {
    "ii_native_naive": "o",
    "ii_native_non_naive": "^"
}

# --------------------------------------------------
# Figure
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharey=False
)

for ax, info in zip(
    axes,
    species_info
):

    # baseline
    ax.scatter(
        baseline[info["x"]],
        baseline[info["y"]],
        color="black",
        marker="s",
        s=140,
        edgecolors="black",
        linewidth=1.2,
        alpha=1
    )

    for exp_type in [
        "ii_native_naive",
        "ii_native_non_naive"
    ]:

        sub = df_final[
            df_final["experiment_type"]
            == exp_type
        ]

        for _, row in sub.iterrows():

            ax.scatter(
                row[info["x"]],
                row[info["y"]],
                color=scenario_colors[row["SC"]],
                marker=markers[exp_type],
                s=120,
                edgecolors="black",
                linewidth=0.8,
                alpha=0.9
            )

    ax.set_title(
        info["title"],
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(r"$IFD/LOF$")

    ax.set_xscale("log")

    ax.grid(
        alpha=0.3,
        linestyle=":"
    )

# --------------------------------------------------
# Axe Y
# --------------------------------------------------

axes[0].set_ylabel(
    "Resource alignment / Risk alignment"
)

# --------------------------------------------------
# Légende scénarios
# --------------------------------------------------

scenario_legend = [
    Line2D(
        [0], [0],
        marker="o",
        color=scenario_colors[sc],
        linestyle="",
        markersize=8,
        label=sc
    )
    for sc in scenarios
]

# --------------------------------------------------
# Légende comportements
# --------------------------------------------------

behaviour_legend = [

    Line2D(
        [0], [0],
        marker="s",
        linestyle="",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=10,
        label="Baseline"
    ),

    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        markersize=10,
        label="Native naive"
    ),

    Line2D(
        [0], [0],
        marker="^",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        markersize=10,
        label="Native non-naive"
    )
]

# --------------------------------------------------
# Affichage des légendes
# --------------------------------------------------

legend_scenarios = axes[2].legend(
    handles=scenario_legend,
    title="Scenario",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    frameon=False
)

axes[2].add_artist(
    legend_scenarios
)

legend_behaviour = axes[2].legend(
    handles=behaviour_legend,
    title="Behaviour",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.55),
    frameon=False
)

# --------------------------------------------------
# Mise en page
# --------------------------------------------------

plt.tight_layout(
    rect=[0, 0, 0.85, 1]
)

plt.show()




# ==========================================================================================
## FIGURE 16: Mesure de l'expansion spatiale en fct du temps
## On considère une cellule occupée si la densité de l,espèce dedans est
## supérieure a un seuil de 1% de sa densité initiale
## et calcule l'aire occupée
## Figure de 4 subplots avec une espece / subplot 
## chaque courbe représente un scenario et le 
## type de trait est naif ou non naif
# ==========================================================================================
from matplotlib.lines import Line2D

scenario_to_sc = {
    "scenario_0000": "SC0",
    "scenario_0001": "SC1",
    "scenario_0002": "SC2",
    "scenario_0003": "SC3",
    "scenario_0004": "SC4",
    "scenario_0005": "SC5",
    "scenario_0006": "SC6",
    "scenario_0007": "SC7",
    "scenario_0008": "SC8",
    "scenario_0009": "SC9",
}



species_list = ["C", "M", "P", "D"]

col_titles = [
    "Caribou",
    "Moose",
    "Wolf",
    "Deer"
]

# Couleurs par scénario
scenarios = sorted(
    df_spread.loc[
        df_spread["experiment_type"] != "baseline",
        "scenario"
    ].unique()
)


# Style naïf / non-naïf
trait_styles = {
    "ii_native_naive": {
        "marker": "o",
        "linestyle": "-"
    },
    "ii_native_non_naive": {
        "marker": "^",
        "linestyle": "--"
    }
}


fig, axes = plt.subplots(
    2, 2,
    figsize=(16,10),
    sharex=True
)

plt.subplots_adjust(right=0.88)

legend_handles = {}

for ax, sp, title in zip(
    axes.ravel(),
    species_list,
    col_titles
):

    sub_sp = df_spread[
        (df_spread["species"] == sp)
        &
        (df_spread["simulation"] == "CMPD")
    ]

    # =====================
    # Baseline (noir)
    # =====================

    baseline = sub_sp[
        sub_sp["experiment_type"] == "baseline"
    ]

    if not baseline.empty:

        ts = (
            baseline
            .groupby("time")["spread"]
            .mean()
            .reset_index()
            .sort_values("time")
        )

        line = ax.plot(
            ts["time"],
            ts["spread"],
            color="black",
            linewidth=3,
            label="Baseline"
        )[0]

        if "Baseline" not in legend_handles:
            legend_handles["Baseline"] = line

    # =====================
    # Scénarios expérimentaux
    # =====================

    exp = sub_sp[
        sub_sp["experiment_type"] != "baseline"
    ]

    for (scenario, trait), sub in exp.groupby(
        ["scenario", "experiment_type"]
    ):

        ts = (
            sub
            .groupby("time")["spread"]
            .mean()
            .reset_index()
            .sort_values("time")
        )

        style = trait_styles.get(
            trait,
            {
                "marker": "o",
                "linestyle": "-"
            }
        )

        color = scenario_colors[
            scenario_to_sc[scenario]
        ]

        short_name = scenario.replace(
            "scenario_",
            ""
        )

        label = (
            f"{short_name} Na"
            if trait == "ii_native_naive"
            else f"{short_name} NNa"
        )

        line = ax.plot(
            ts["time"],
            ts["spread"],
            color=color,
            linestyle=style["linestyle"],
            linewidth=1.5,
            alpha=0.8
        )[0]

        if label not in legend_handles:
            legend_handles[label] = line

    ax.set_title(title, fontsize=14)

# Axes
for ax in axes[1, :]:
    ax.set_xlabel("Time")

for ax in axes[:, 0]:
    ax.set_ylabel("Proportion of occupied domain")

# ======================
# Légende couleurs (scénarios)
# ======================

scenario_handles = [
    Line2D(
        [0], [0],
        color=scenario_colors[
            scenario_to_sc[sc]
        ],
        lw=2,
        label=sc.replace("scenario_", "SC")
    )
    for sc in scenarios
]

# ======================
# Légende type de trait
# ======================

trait_handles = [
    Line2D(
        [0], [0],
        color="black",
        lw=2,
        linestyle="-",
        label="Naïf"
    ),
    Line2D(
        [0], [0],
        color="black",
        lw=2,
        linestyle="--",
        label="Non naïf"
    ),
    Line2D(
        [0], [0],
        color="black",
        lw=3,
        linestyle="-",
        label="Baseline"
    )
]

fig.legend(
    handles=scenario_handles,
    title="Scenario",
    loc="upper right",
    bbox_to_anchor=(0.99, 0.98),
    fontsize=8,
    title_fontsize=9,
    frameon=False
)

fig.legend(
    handles=trait_handles,
    title="Trait type",
    loc="upper right",
    bbox_to_anchor=(0.99, 0.55),
    fontsize=8,
    title_fontsize=9,
    frameon=False
)

plt.tight_layout(
    rect=[0, 0, 0.88, 1]
)

plt.show()





# ==========================================================================================
## FIGURE 17: Autre figure: Delta spread rate pour chaque espèce
# ==========================================================================================

from matplotlib.lines import Line2D

col_titles = [
    "Caribou",
    "Moose",
    "Wolf",
    "Deer"
]


naive = df_spread[
    df_spread["experiment_type"]
    == "ii_native_naive"
]

non_naive = df_spread[
    df_spread["experiment_type"]
    == "ii_native_non_naive"
]

delta = naive.merge(
    non_naive,
    on=[
        "species",
        "scenario",
        "time"
    ],
    suffixes=(
        "_naive",
        "_non_naive"
    )
)

delta["dSpread"] = (
    delta["spread_non_naive"]
    -
    delta["spread_naive"]
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,8),
    sharex=True
)

for ax, sp, title in zip(
    axes.ravel(),
    ["C","M","P","D"],
    col_titles
):

    sub_sp = delta[
        delta["species"] == sp
    ]

    for scenario, sub in sub_sp.groupby("scenario"):

        sc = sub["SC_naive"].iloc[0]

        ax.plot(
            sub["time"],
            sub["dSpread"],
            color=scenario_colors[sc],
            lw=2
        )


    ax.axhline(
        0,
        color="black",
        ls="--"
    )

    ax.set_title(title)

for ax in axes[1,:]:
    ax.set_xlabel("Time")

for ax in axes[:,0]:
    ax.set_ylabel(
        r"$\Delta$ spread = non-naive $-$ naive"
    )

from matplotlib.lines import Line2D

scenario_handles = [
    Line2D(
        [0], [0],
        color=scenario_colors[sc],
        lw=2,
        label=sc
    )
    for sc in sorted(
        delta["SC_naive"].unique(),
        key=lambda x: int(x.replace("SC",""))
    )
]

legend = fig.legend(
    handles=scenario_handles,
    title="Scenario",
    loc="center right",
    bbox_to_anchor=(0.98, 0.5),
    frameon=True
)

plt.subplots_adjust(right=0.80)

plt.show()




## ======================================================================
## FIGURE 18: Time to reach equilibrium ou REGARDER LA VITESSE DE CONVERGENCE
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





# ==========================================================================================
## FIGURE 19: Autre figure: Delta spread rate pour chaque espèce


# ==========================================================================================


## ====================================
# ====================================
# FIGURE
#
# Axe X : c_ratio = C_IFD / C_LOF (échelle logarithmique)
#
# Plus on est à gauche :
#     -> la contribution de l'IFD est faible
#     -> le comportement est davantage influencé par LOF
#
# Plus on est à droite :
#     -> la contribution de l'IFD est forte relativement à LOF
#
# Axe Y : alignement avec la ressource /
#          alignement avec le prédateur
#
# Valeurs > 1 :
#     -> l'espèce est davantage alignée avec les ressources
#        qu'avec le prédateur
#
# Valeurs < 1 :
#     -> l'espèce est davantage alignée avec le prédateur
#        qu'avec les ressources
#
# Plus on monte :
#     -> l'influence relative des ressources augmente
#
# Plus on descend :
#     -> l'influence relative du prédateur augmente
#
# Chaque point représente un scénario.
# ====================================

plt.scatter(
    df_final["c_ratio"],
    df_final["C_IFD_ratio_P"],
    label="Caribou"
)

plt.scatter(
    df_final["c_ratio"],
    df_final["M_IFD_ratio_P"],
    label="Moose"
)

plt.scatter(
    df_final["c_ratio"],
    df_final["D_IFD_ratio_P"],
    label="Deer"
)

plt.xlabel(
    r"$C_{IFD}/C_{LOF}$ (log scale)"
)

plt.ylabel(
    "Resource alignment / Predator alignment"
)

plt.legend()

plt.xscale("log")

plt.show()




# =================== Autres metriquess ======================

cmpd["C_tradeoff_P"] = (
    cmpd["C-IFDresource_ALIGN"]
    - cmpd["C-riskP_ALIGN"]
)

cmpd["C_tradeoff_PM"] = (
    cmpd["C-IFDresource_ALIGN"]
    - cmpd["C-riskPM_ALIGN"]
)

cmpd["C_tradeoff_PMD"] = (
    cmpd["C-IFDresource_ALIGN"]
    - cmpd["C-riskPMD_ALIGN"]
)

tmax = cmpd["time"].max()

cmpd_final = cmpd[
    cmpd["time"] == tmax
].copy()


pivot = (
    cmpd_final[
        [
            "SC",
            "experiment_type",
            "C_tradeoff_PMD"
        ]
    ]
    .pivot(
        index="SC",
        columns="experiment_type",
        values="C_tradeoff_PMD"
    )
)

pivot = (
    cmpd_final[
        [
            "SC",
            "experiment_type",
            "C-IFDresource_ALIGN"
        ]
    ]
    .pivot(
        index="SC",
        columns="experiment_type",
        values="C-IFDresource_ALIGN"
    )
)

pivot["delta"] = (
    pivot["ii_native_non_naive"]
    -
    pivot["ii_native_naive"]
)

print(pivot)

delta_C = (
    cmpd_final
    .pivot(
        index="SC",
        columns="experiment_type",
        values="C-IFDresource_ALIGN"
    )
)

delta_C["delta"] = (
    delta_C["ii_native_non_naive"]
    -
    delta_C["ii_native_naive"]
)

print(delta_C["delta"].describe())


pivot = (
    cmpd_final[
        [
            "SC",
            "experiment_type",
            "C-riskPMD_ALIGN"
        ]
    ]
    .pivot(
        index="SC",
        columns="experiment_type",
        values="C-riskPMD_ALIGN"
    )
)

pivot["delta"] = (
    pivot["ii_native_non_naive"]
    -
    pivot["ii_native_naive"]
)

print(pivot["delta"].describe())




pivot = (
    cmpd_final[
        [
            "SC",
            "experiment_type",
            "C_tradeoff_PMD"
        ]
    ]
    .pivot(
        index="SC",
        columns="experiment_type",
        values="C_tradeoff_PMD"
    )
)

pivot["delta"] = (
    pivot["ii_native_non_naive"]
    -
    pivot["ii_native_naive"]
)

print(pivot["delta"].describe())