import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


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

df_wide["SC"] = (
    df_wide["scenario"]
    .map(scenario_map)
)

df_wide["SC_label"] = (
    df_wide["simulation"]
    + "_"
    + df_wide["SC"]
)




unique_sc = sorted(
    df_wide["SC"].unique()
)

cmap = plt.cm.tab20

scenario_colors = {

    sc: cmap(
        i / max(1, len(unique_sc)-1)
    )

    for i, sc
    in enumerate(unique_sc)
}


experiment_styles = {

    "baseline": {
        "linestyle": "-",
        "alpha": 1.0,
        "lw": 3
    },

    "ii_native_naive": {
        "linestyle": "--",
        "alpha": 0.8,
        "lw": 2
    },

    "ii_native_non_naive": {
        "linestyle": ":",
        "alpha": 0.8,
        "lw": 2
    }
}


baseline = df_wide[
    df_wide["experiment_type"] == "baseline"
]

baseline_cmp = baseline[
    baseline["simulation"] == "CMP"
]

baseline_cmpd = baseline[
    baseline["simulation"] == "CMPD"
]

alternative = df_wide[
    df_wide["experiment_type"] != "baseline"
]

alternative_cmp = alternative[
    alternative["simulation"] == "CMP"
]

alternative_cmpd = alternative[
    alternative["simulation"] == "CMPD"
]







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

# selected_times = [
#     times[0],
#     times[len(times)//4],
#     times[len(times)//2],
#     times[3*len(times)//4],
#     times[-1]
# ]

selected_times = np.unique(
    times[
        np.linspace(
            0,
            len(times)-1,
            10,
            dtype=int
        )
    ]
)



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

alphas = np.linspace(
    0.3,
    1.0,
    len(selected_times)
)

time_legend = [

    Line2D(
        [0],[0],
        marker="o",
        color="grey",
        alpha=a,
        linestyle="",
        markersize=10,
        label=f"t={t}"
    )

    for a, t in zip(
        alphas,
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

    for (sim, sc, exp_type), sub in (
    df_wide.groupby(
        ["simulation",
         "SC",
         "experiment_type"]
    )
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
            color=scenario_colors[sc],
            linestyle=experiment_styles[exp_type]["linestyle"],
            linewidth=experiment_styles[exp_type]["lw"],
            alpha=experiment_styles[exp_type]["alpha"]
        )

        ax.text(
            sub["time"].iloc[-1],
            sub[pair].iloc[-1],
            sub["SC_label"].iloc[0],
            fontsize=7
        )

        #ax.scatter(
        #    sub["time"],
        #    sub[pair],
        #    color=scenario_colors[sc],
        #    s=15
        #)

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

# df_delta = alternative.merge(
#     baseline,
#     on=merge_cols,
#     suffixes=("_alt", "_base"),
#     validate="one_to_one"
# )


df_delta = alternative.merge(
    baseline.drop(columns=["experiment_type"]),
    on=[
        "scenario",
        "simulation",
        "time"
    ],
    suffixes=("_alt", "_base"),
    validate="many_to_one"
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

    ("ii_native_naive", "tab:orange"),

    ("ii_native_non_naive", "tab:red")

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
print(df_dist.describe())


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



# color_map = {
#     ("CMP", "baseline"): "tab:blue",
#     ("CMP", "alternative"): "tab:cyan",
#     ("CMPD", "baseline"): "tab:red",
# }


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

    for (sim, exp_type, sc), sub_all in (
    df_wide.groupby(
        ["simulation",
         "experiment_type",
         "SC"]
    )
):

        for scen, sub in (
            sub_all.groupby("scenario")
        ):

            sub = sub.sort_values("time")

            line = ax.plot(
                sub["time"],
                sub[var],
                color=scenario_colors[sc],
                linestyle=experiment_styles[exp_type]["linestyle"],
                linewidth=experiment_styles[exp_type]["lw"],
                alpha=experiment_styles[exp_type]["alpha"]
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
## =====================================================================

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

# colors = {
#     "CMP": "tab:blue",
#     "CMPD": "tab:red"
# }

linestyles = {
    "P": "-",
    "PM": "--",
    "PMD": ":"
}


for (pset, sim, scen), sub in df_wide.groupby(
    ["parameter_set", "simulation", "scenario"]
):


    exp_type = sub["experiment_type"].iloc[0]

    sc = sub["SC"].iloc[0]

    sub = sub.sort_values("time")

    lw = (
        3 if pset == "i"
        else 1)
    
    ax.plot(
        sub["time"],
        sub["C_ratio_P"],
        color=scenario_colors[sc],
        linestyle="-",
        linewidth=experiment_styles[exp_type]["lw"],
        alpha=experiment_styles[exp_type]["alpha"]
    )

    ax.plot(
        sub["time"],
        sub["C_ratio_PM"],
        color=scenario_colors[sc],
        linestyle="--",
        linewidth=experiment_styles[exp_type]["lw"],
        alpha=experiment_styles[exp_type]["alpha"]
    )

    ax.plot(
        sub["time"],
        sub["C_ratio_PMD"],
        color=scenario_colors[sc],
        linestyle=":",
        linewidth=experiment_styles[exp_type]["lw"],
        alpha=experiment_styles[exp_type]["alpha"]
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

# colors = {
#     ("CMP", "baseline"): "tab:blue",
#     ("CMP", "alternative"): "tab:cyan",
#     ("CMPD", "baseline"): "tab:red",
#     ("CMPD", "alternative"): "tab:orange",
# }


for ax, var, title in zip(axes, variables, titles):

    for (sim, exp_type, scen, sc), sub in (
    df_wide.groupby(
        [
            "simulation",
            "experiment_type",
            "scenario",
            "SC"
        ]
    )
):

        sub = sub.sort_values("time")

        ax.plot(
            sub["time"],
            sub[var],
            color=scenario_colors[sc],
            linestyle=experiment_styles[exp_type]["linestyle"],
            linewidth=experiment_styles[exp_type]["lw"],
            alpha=experiment_styles[exp_type]["alpha"]
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

scenario_order = (
    df_final[
        ["SC","c_ratio"]
    ]
    .drop_duplicates()
    .sort_values("c_ratio")
)

ratio_colors = {
    sc: plt.cm.viridis(
        i / max(1, len(scenario_order)-1)
    )
    for i, sc in enumerate(
        scenario_order["SC"]
    )
}


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
        # offset = {
        #     "CMP": -0.01,
        #     "CMPD": 0.01
        #     }

        point_colors = [
            ratio_colors[sc]
            for sc in sub["SC"]
        ]

        ax.scatter(
            sub["c_ratio"],
            sub[ratio],
            marker=marker,
            c=point_colors,
            s=100,
            alpha=0.8,
            edgecolors="black",
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




# ================== TEST POUR REMPLACER FIG 1 
pairs = [
    "C-M_BC",
    "C-P_BC",
    "M-P_BC"
]

ratio_values = (
    df_wide[
        ["SC", "c_ratio"]
    ]
    .drop_duplicates()
    .sort_values("c_ratio")
)

norm = plt.Normalize(
    df_wide["c_ratio"].min(),
    df_wide["c_ratio"].max()
)

ratio_colors = {

    sc: plt.cm.viridis(
        norm(c_ratio)
    )

    for sc, c_ratio in
    ratio_values[
        ["SC", "c_ratio"]
    ].values
}


markers = {
    "baseline": "o",
    "ii_native_naive": "^",
    "ii_native_non_naive": "s"
}

linestyles = {
    "CMP": "-",
    "CMPD": "--"
}


fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(15,5),
    sharey=True
)

for ax, pair in zip(axes, pairs):

    for (sim, sc, exp_type), sub in df_wide.groupby(
    ["simulation", "SC", "experiment_type"]
):

        sub = sub.sort_values("time")

        color = ratio_colors[sc]

        ax.plot(
            sub["time"],
            sub[pair],
            color=color,
            linestyle=linestyles[sim],
            alpha=0.6,
            lw=1.5
        )

        ax.scatter(
            sub["time"],
            sub[pair],
            color=color,
            marker=markers[exp_type],
            s=40
        )

        ax.text(
            sub["time"].max(),
            sub[pair].iloc[-1],
            sc,
            fontsize=6,
            alpha=0.8
        )

    ax.set_title(
        pair.replace("_BC", "")
    )

    ax.set_xlabel(
        "Temps"
    )

axes[0].set_ylabel(
    "Bhattacharyya coefficient"
)


legend_exp = [

    Line2D(
        [0],[0],
        marker="o",
        color="black",
        linestyle="",
        label="baseline"
    ),

    Line2D(
        [0],[0],
        marker="^",
        color="black",
        linestyle="",
        label="ii_native_naive"
    ),

    Line2D(
        [0],[0],
        marker="s",
        color="black",
        linestyle="",
        label="ii_native_non_naive"
    )
]



sm = plt.cm.ScalarMappable(
    norm=norm,
    cmap="viridis"
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    shrink=0.8
)

cbar.set_label(
    r"$C_{IFD}/C_{LOF}$"
)

axes[-1].legend(
    handles=legend_exp,
    loc="upper left"
)

plt.tight_layout()
plt.show()



# ====================== Get the maximizing parameters 
# to use for scenario iii)

best_C = df_final.loc[
    df_final["C_ratio_P"].idxmax()
]

print(best_C[
    [
        "scenario",
        "simulation",
        "c_ratio",
        "C_ratio_P",
        "c_ifd",
        "c_lof",
        "m_ifd",
        "m_lof",
        "p_ifd",
        "p_lof"
    ]
])

best_M = df_final.loc[
    df_final["M_ratio_P"].idxmax()
]

best_M = df_final.loc[
    df_final["M_ratio_P"].idxmax()
]

print(
    best_M[
        [
            "scenario",
            "simulation",
            "M_ratio_P",
            "c_ratio",
            "c_ifd",
            "c_lof",
            "m_ifd",
            "m_lof",
            "p_ifd",
            "p_lof"
        ]
    ]
)


for sim in ["CMP", "CMPD"]:

    sub = df_final[
        df_final["simulation"] == sim
    ]

    best = sub.loc[
        sub["C_ratio_P"].idxmax()
    ]

    print("\n", sim)
    print(
        best[
            [
                "scenario",
                "C_ratio_P",
                "c_ratio"
            ]
        ]
    )


## Pour le carbou:
# ================
# C
# ================
# pair
# scenario         scenario_0002
# simulation                CMPD
# c_ratio               3.610343
# C_tradeoff            -0.29947
# C-Phi_C_ALIGN          0.49289
# C-riskP_ALIGN          0.79236


## Pour l'orignal
# ================
# M
# ================
# pair
# scenario         scenario_0009
# simulation                CMPD
# c_ratio               1.169192
# M_tradeoff           -1.414257
# M-Phi_M_ALIGN        -0.538927
# M-riskP_ALIGN         0.875329


df_final["M_score"] = (
    df_final["M-Phi_M_ALIGN"]
    -
    df_final["M-riskP_ALIGN"]
)

best_M = df_final.loc[
    df_final["M_score"].idxmax()
]

print(
    best_M[
        [
            "scenario",
            "simulation",
            "c_ratio",
            "M_score",
            "M-Phi_M_ALIGN",
            "M-riskP_ALIGN"
        ]
    ]
)

plt.scatter(
    df_final["c_ratio"],
    df_final["C_ratio_P"],
    label="Caribou"
)

plt.scatter(
    df_final["c_ratio"],
    df_final["M_ratio_P"],
    label="Moose"
)

plt.legend()
plt.xscale("log")
plt.show()


plt.figure(figsize=(7,7))

for sp in ["C", "M", "D"]:

    plt.scatter(
        df_final[f"{sp}-riskP_ALIGN"],
        df_final[f"{sp}-Phi_{sp}_ALIGN"],
        label=sp,
        s=80
    )

plt.axvline(0,color="k",ls="--")
plt.axhline(0,color="k",ls="--")

plt.xlabel("Alignement prédateur")
plt.ylabel("Alignement ressource")

plt.legend()
plt.show()


df_final = (
    df_wide[
        df_wide["time"] == tmax
    ]
    .copy()
)


df_final["C_tradeoff"] = (
    df_final["C-Phi_C_ALIGN"]
    - df_final["C-riskP_ALIGN"]
)

df_final["M_tradeoff"] = (
    df_final["M-Phi_M_ALIGN"]
    - df_final["M-riskP_ALIGN"]
)

df_final["D_tradeoff"] = (
    df_final["D-Phi_D_ALIGN"]
    - df_final["D-riskP_ALIGN"]
)


df_final = (
    df_wide[df_wide["time"] == tmax]
    .copy()
)


species = [
    ("C", "Caribou"),
    ("M", "Moose"),
    ("D", "Deer")
]

fig, axes = plt.subplots(
    1, 3,
    figsize=(15,5),
    sharex=True,
    sharey=True
)

for ax, (sp, name) in zip(axes, species):

    sc = ax.scatter(
        df_final[f"{sp}-riskP_ALIGN"],
        df_final[f"{sp}-Phi_{sp}_ALIGN"],
        c=df_final["c_ratio"],
        cmap="viridis",
        s=150,
        edgecolors="black"
    )

    ax.axvline(0, color="k", ls="--")
    ax.axhline(0, color="k", ls="--")

    ax.set_title(name)

    ax.set_xlabel(
        "Alignement prédateur"
    )

axes[0].set_ylabel(
    "Alignement ressource"
)

cbar = plt.colorbar(
    sc,
    ax=axes
)

cbar.set_label(
    r"$C_{IFD}/C_{LOF}$"
)

plt.tight_layout()
plt.show()


for sp in ["C","M","D"]:

    df_final[f"{sp}_tradeoff"] = (
        df_final[f"{sp}-Phi_{sp}_ALIGN"]
        -
        df_final[f"{sp}-riskP_ALIGN"]
    )

    best = df_final.loc[
        df_final[f"{sp}_tradeoff"].idxmax()
    ]

    print("\n================")
    print(sp)
    print("================")

    print(
        best[
            [
                "scenario",
                "simulation",
                "c_ratio",
                f"{sp}_tradeoff",
                f"{sp}-Phi_{sp}_ALIGN",
                f"{sp}-riskP_ALIGN"
            ]
        ]
    )
