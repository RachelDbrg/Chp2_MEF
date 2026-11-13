import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# Domaine spatial 2D
# ==================================================

x = np.linspace(0, 10, 150)
y = np.linspace(0, 10, 150)

X, Y = np.meshgrid(x, y)


sigma = 0.5

# ==================================================
# Scénarios
# ==================================================

scenarios = [
    {
        "titre": "Aucun chevauchement",
        "mu_R": (2, 2),
        "mu_C": (5, 5),
        "mu_P": (8, 8)
    },
    {
        "titre": "Ressource-Proie",
        "mu_R": (2, 2),
        "mu_C": (2, 2),
        "mu_P": (8, 8)
    },
    {
        "titre": "Proie-Prédateur",
        "mu_R": (2, 2),
        "mu_C": (8, 8),
        "mu_P": (8, 8)
    },
    {
        "titre": "Chevauchement des 3",
        "mu_R": (5, 5),
        "mu_C": (5, 5),
        "mu_P": (5, 5)
    }
]

# ==================================================
# Figure
# ==================================================

fig = plt.figure(figsize=(20, 5))

for k, s in enumerate(scenarios):

    # ------------------------------------------
    # Distributions spatiales
    # ------------------------------------------

    R = np.exp(
        -(
            (X - s["mu_R"][0])**2 +
            (Y - s["mu_R"][1])**2
        ) / (2 * sigma**2)
    )

    C = np.exp(
        -(
            (X - s["mu_C"][0])**2 +
            (Y - s["mu_C"][1])**2
        ) / (2 * sigma**2)
    )

    P = np.exp(
        -(
            (X - s["mu_P"][0])**2 +
            (Y - s["mu_P"][1])**2
        ) / (2 * sigma**2)
    )

    # ------------------------------------------
    # Score spatial local
    # ------------------------------------------

    J = (R * C) / (1 + P * C)

    # ------------------------------------------
    # Plot 3D
    # ------------------------------------------

    ax = fig.add_subplot(2, 2, k + 1, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        J,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    # Ressource
    ax.contour(
        X, Y, R,
        zdir='z',
        offset=1,
        colors='green',
        linewidths=2
    )

    # Proie
    ax.contour(
        X, Y, C,
        zdir='z',
        offset=1,
        colors='blue',
        linewidths=2
    )

    # Prédateur
    ax.contour(
        X, Y, P,
        zdir='z',
        offset=1,
        colors='red',
        linewidths=2
    )

    ax.set_title(s["titre"])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("J")
    ax.set_zlim(0, 1)

    fig.colorbar(
    surf,
    ax=ax,
    shrink=0.6,
    pad=0.05
)

    ax.view_init(elev=30, azim=135)

plt.suptitle(
    r"$J(x,y)=\frac{R(x,y)C(x,y)}{1+P(x,y)C(x,y)}$",
    fontsize=16
)

plt.tight_layout()
plt.show()