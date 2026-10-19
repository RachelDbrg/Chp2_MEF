import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

## Definit le .csv dans lequel on a regroupé les 5 snapshots pour tous les scenarios testés
output_csv = "/users/rdubourg/Documents/Chp2_MEF/figures/overlap_strategy_snapshots.csv"

## lit le .csv
df = pd.read_csv(
    output_csv
)

print(df.head())

## Definition du type de stratégie based on les coefficiens IFD/LOF logs dans les parametres
# Pour le caribou
df["c_ratio"] = (
    df["c_ifd"]
    / df["c_lof"]
)

# Pour l'orignal
df["m_ratio"] = (
    df["m_ifd"]
    / df["m_lof"]
)

markers = {
    0: "o",
    25: "s",
    50: "^",
    75: "D",
    100: "*"
}


# -------------------------------------------------------------------------------------------------

## graphiques pour le caribou

plt.figure(figsize=(8,6))

sc = plt.scatter(
    df["c_lof"],
    df["overlap_CP"],
    c=df["time"],
    cmap="viridis",
    s=120,
    edgecolors="black"
)

plt.xlabel(r"$C_{IFD}/C_{LOF}$")
plt.ylabel("Overlap CP (%)")

cbar = plt.colorbar(sc)
cbar.set_label("Time")

plt.tight_layout()
plt.show()



# -------------------------------------------------------------------------------------------------

## graphiques pour l'orignal

plt.figure(figsize=(8,6))

sc = plt.scatter(
    df["m_lof"],
    df["overlap_MP"],
    c=df["time"],
    cmap="viridis",
    s=120,
    edgecolors="black"
)

plt.xlabel(r"$M_{IFD}/M_{LOF}$")
plt.ylabel("Overlap mP (%)")

cbar = plt.colorbar(sc)
cbar.set_label("Time")

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------------

## graphiques pour orignal vs. caribou

plt.figure(figsize=(8,6))

sc = plt.scatter(
    df["c_ifd"],
    df["m_ifd"],
    c=df["overlap_CM"],
    cmap="viridis",
    s=120,
    edgecolors="black"
)

plt.xlabel("c_ifd")
plt.ylabel("m_ifd")

cbar = plt.colorbar(sc)
cbar.set_label("overlap_CM")

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt

markers = {
    0: "o",
    25: "s",
    50: "^",
    75: "D",
    100: "*"
}

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10),
    sharex=False,
    sharey=False
)

# -----------------------------
# C ratio vs overlap_CM
# -----------------------------

ax = axes[0, 0]

for t in sorted(df["time"].unique()):

    sub = df[df["time"] == t]

    ax.scatter(
        sub["c_lof"],
        sub["overlap_CM"],
        marker=markers[t],
        s=120,
        alpha=0.8,
        label=f"t={t}"
    )

ax.set_xlabel(r"$C_{LOF}$")
ax.set_ylabel("Overlap CM (%)")
ax.set_title("Caribou strategy vs CM overlap")

# -----------------------------
# C ratio vs overlap_CP
# -----------------------------

ax = axes[0, 1]

for t in sorted(df["time"].unique()):

    sub = df[df["time"] == t]

    ax.scatter(
        sub["c_lof"],
        sub["overlap_CP"],
        marker=markers[t],
        s=120,
        alpha=0.8
    )

ax.set_xlabel(r"$C_{LOF}$")
ax.set_ylabel("Overlap CP (%)")
ax.set_title("Caribou strategy vs CP overlap")

# -----------------------------
# M ratio vs overlap_CM
# -----------------------------

ax = axes[1, 0]

for t in sorted(df["time"].unique()):

    sub = df[df["time"] == t]

    ax.scatter(
        sub["m_lof"],
        sub["overlap_CM"],
        marker=markers[t],
        s=120,
        alpha=0.8
    )

ax.set_xlabel(r"$M_{LOF}$")
ax.set_ylabel("Overlap CM (%)")
ax.set_title("Moose strategy vs CM overlap")

# -----------------------------
# M ratio vs overlap_MP
# -----------------------------

ax = axes[1, 1]

for t in sorted(df["time"].unique()):

    sub = df[df["time"] == t]

    ax.scatter(
        sub["m_lof"],
        sub["overlap_MP"],
        marker=markers[t],
        s=120,
        alpha=0.8
    )

ax.set_xlabel(r"$M_{LOF}$")
ax.set_ylabel("Overlap MP (%)")
ax.set_title("Moose strategy vs MP overlap")

# -----------------------------
# Légende commune
# -----------------------------

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Time",
    loc="upper center",
    ncol=5
)

plt.tight_layout()

plt.show()














#
# Couleurs de base
colors = {
    "CM": "blue",
    "CP": "red",
    "MP": "green"
}

fig, axes = plt.subplots(
    1, 3,
    figsize=(15,5),
    sharey=True
)

pairs = [
    ("overlap_CM", "Blues"),
    ("overlap_CP", "Reds"),
    ("overlap_MP", "Greens")
]

for ax, (ov, cmap) in zip(
    axes,
    pairs
):

    sc = ax.scatter(
        df["c_lof"],
        df[ov],
        c=df["time"],
        cmap=cmap,
        s=80
    )

    ax.set_title(
        ov.replace(
            "overlap_",
            ""
        )
    )

    ax.set_xlabel(
        r"$C_{IFD}/C_{LOF}$"
    )

axes[0].set_ylabel(
    "Overlap (%)"
)

cbar = fig.colorbar(
    sc,
    ax=axes,
    shrink=0.8
)

cbar.set_label(
    "Time step"
)

plt.tight_layout()
#plt.show()


plt.hist(df["c_lof"], bins=20)
plt.show()

print(df["c_lof"].min())
print(df["c_lof"].max())

print(df["overlap_CM"].min())
print(df["overlap_CM"].max())

print(df["overlap_CP"].min())
print(df["overlap_CP"].max())

print(df["overlap_MP"].min())
print(df["overlap_MP"].max())


df_scen = (
    df.groupby("scenario")
      .agg({
          "c_lof":"first",
          "m_lof":"first",
          "p_lof":"first",
          "overlap_CM":"mean",
          "overlap_CP":"mean",
          "overlap_MP":"mean"
      })
      .reset_index()
)

print(
    "corr(c_lof, overlap_CM) =",
    df_scen["c_lof"].corr(
        df_scen["overlap_CM"]
    )
)

print(
    "corr(c_lof, overlap_CP) =",
    df_scen["c_lof"].corr(
        df_scen["overlap_CP"]
    )
)

print(
    "corr(c_lof, overlap_MP) =",
    df_scen["c_lof"].corr(
        df_scen["overlap_MP"]
    )
)




df_scen = (
    df.groupby("scenario")
      .agg({
          "c_lof":"first",
          "overlap_CM":"mean",
          "overlap_CP":"mean",
          "overlap_MP":"mean"
      })
      .reset_index()
)


fig, axes = plt.subplots(
    1, 3,
    figsize=(15,5),
    sharex=True
)

overlaps = [
    "overlap_CM",
    "overlap_CP",
    "overlap_MP"
]

for ax, ov in zip(axes, overlaps):

    corr = df_scen["c_lof"].corr(
        df_scen[ov]
    )

    ax.scatter(
        df_scen["c_lof"],
        df_scen[ov],
        s=100
    )

    ax.set_title(
        f"{ov}\nr={corr:.2f}"
    )

    ax.set_xlabel("c_lof")

axes[0].set_ylabel("Mean overlap")

plt.tight_layout()
plt.show()