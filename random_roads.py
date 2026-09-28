import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ---------------------------------------
# Parameters
# ---------------------------------------

road_width = 0.001
target_coverage = 0.001
max_attempts = 10000
grid_size = 500

roads = []

# Coverage grid
occupied = np.zeros((grid_size, grid_size), dtype=bool)
coverage = 0.0

# ---------------------------------------
# Utilities
# ---------------------------------------

def road_polygon(x0, y0, x1, y1, width):

    dx = x1 - x0
    dy = y1 - y0

    L = np.hypot(dx, dy)

    if L == 0:
        return None

    nx = -dy / L
    ny = dx / L

    w = width / 2

    return np.array([
        [x0 + w * nx, y0 + w * ny],
        [x0 - w * nx, y0 - w * ny],
        [x1 - w * nx, y1 - w * ny],
        [x1 + w * nx, y1 + w * ny]
    ])


def segment_distance(a, b, c, d):
    """
    Approximate minimum distance between two segments.
    """

    pts1 = np.linspace(a, b, 20)
    pts2 = np.linspace(c, d, 20)

    return np.min([
        np.linalg.norm(p - q)
        for p in pts1
        for q in pts2
    ])


def add_road_to_grid(x0, y0, x1, y1, width, occupied):

    n = 200

    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)

    radius = int(np.ceil(width * grid_size / 2))

    for x, y in zip(xs, ys):

        i = int(x * (grid_size - 1))
        j = int(y * (grid_size - 1))

        i0 = max(0, i - radius)
        i1 = min(grid_size, i + radius + 1)

        j0 = max(0, j - radius)
        j1 = min(grid_size, j + radius + 1)

        occupied[j0:j1, i0:i1] = True


# ---------------------------------------
# Generation
# ---------------------------------------

attempt = 0

while coverage < target_coverage and attempt < max_attempts:

    attempt += 1

    x0, y0 = np.random.rand(2)

    theta = np.random.uniform(0, np.pi)
    length = np.random.uniform(0.2, 0.8)

    x1 = x0 + length * np.cos(theta)
    y1 = y0 + length * np.sin(theta)

    # Keep road inside domain
    if not (0 <= x1 <= 1 and 0 <= y1 <= 1):
        continue

    valid = True

    for r in roads:

        d = segment_distance(
            np.array([x0, y0]),
            np.array([x1, y1]),
            np.array([r["x0"], r["y0"]]),
            np.array([r["x1"], r["y1"]])
        )

        if d < road_width:
            valid = False
            break

    if not valid:
        continue

    roads.append({
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1
    })

    add_road_to_grid(
        x0, y0,
        x1, y1,
        road_width,
        occupied
    )

    coverage = occupied.mean()

    print(
        f"Roads = {len(roads):3d}   "
        f"Coverage = {100*coverage:6.2f}%"
    )

# ---------------------------------------
# Results
# ---------------------------------------

print("\nGeneration complete")
print(f"Number of roads : {len(roads)}")
print(f"Coverage        : {100*coverage:.2f}%")
print(f"Attempts        : {attempt}")



# ---------------------------------------
# Road coordinates - saved in a .txt file
# ---------------------------------------

with open("road_bounds.txt", "w") as f:

    f.write("Road xmin xmax ymin ymax\n")

    for i, r in enumerate(roads):

        xmin = min(r["x0"], r["x1"])
        xmax = max(r["x0"], r["x1"])
        ymin = min(r["y0"], r["y1"])
        ymax = max(r["y0"], r["y1"])

        f.write(
            f"{i+1} {xmin:.4f} {xmax:.4f} "
            f"{ymin:.4f} {ymax:.4f}\n"
        )


        
# ---------------------------------------
# Plot
# ---------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    "k"
)

for r in roads:

    poly = road_polygon(
        r["x0"],
        r["y0"],
        r["x1"],
        r["y1"],
        road_width
    )

    ax.add_patch(
        Polygon(
            poly,
            closed=True,
            facecolor="gray",
            edgecolor="none"
        )
    )

ax.set_aspect("equal")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("roads.png", dpi=300)


### Write the .champs to be plugged in the def_LF.champs

# ---------------------------------------
# Export obstacle definitions
# ---------------------------------------

with open("Linear_Features.champs", "w") as f:

    for i, r in enumerate(roads, start=1):

        xmin = min(r["x0"], r["x1"])
        xmax = max(r["x0"], r["x1"])
        ymin = min(r["y0"], r["y1"])
        ymax = max(r["y0"], r["y1"])

        f.write(f"# ROAD {i}\n")

        # BAS
        f.write(
            f"scalaire                    fct_obstacle_bas_{i}      "
            f"if(y={ymin:.6f} & x>={xmin:.6f} & x<={xmax:.6f}, 1, 0)\n"
        )
        f.write(
            f"critere_fct_caracteristique obstacle_bas_{i}          "
            f"[fct_obstacle_bas_{i}, 0]\n"
        )
        f.write(
            f"cree_entite_avec_critere    bas_obstacle_{i}          "
            f"ENTITE_TOUTES_COMPOSANTES(obstacle_bas_{i})\n\n"
        )

        # HAUT
        f.write(
            f"scalaire                    fct_obstacle_haut_{i}     "
            f"if(y={ymax:.6f} & x>={xmin:.6f} & x<={xmax:.6f}, 1, 0)\n"
        )
        f.write(
            f"critere_fct_caracteristique obstacle_haut_{i}         "
            f"[fct_obstacle_haut_{i}, 0]\n"
        )
        f.write(
            f"cree_entite_avec_critere    haut_obstacle_{i}         "
            f"ENTITE_TOUTES_COMPOSANTES(obstacle_haut_{i})\n\n"
        )

        # GAUCHE
        f.write(
            f"scalaire                    fct_obstacle_gauche_{i}   "
            f"if(x={xmin:.6f} & y>={ymin:.6f} & y<={ymax:.6f}, 1, 0)\n"
        )
        f.write(
            f"critere_fct_caracteristique obstacle_gauche_{i}       "
            f"[fct_obstacle_gauche_{i}, 0]\n"
        )
        f.write(
            f"cree_entite_avec_critere    gauche_obstacle_{i}       "
            f"ENTITE_TOUTES_COMPOSANTES(obstacle_gauche_{i})\n\n"
        )

        # DROITE
        f.write(
            f"scalaire                    fct_obstacle_droite_{i}   "
            f"if(x={xmax:.6f} & y>={ymin:.6f} & y<={ymax:.6f}, 1, 0)\n"
        )
        f.write(
            f"critere_fct_caracteristique obstacle_droite_{i}       "
            f"[fct_obstacle_droite_{i}, 0]\n"
        )
        f.write(
            f"cree_entite_avec_critere    droite_obstacle_{i}       "
            f"ENTITE_TOUTES_COMPOSANTES(obstacle_droite_{i})\n\n"
        )

        # OBSTACLE ENTIER (INTERIEUR)
        f.write("# ------------------------------------------------------------\n")
        f.write(f"# Obstacle entier {i}\n")
        f.write("# ------------------------------------------------------------\n")

        f.write(
            f"scalaire                    fct_obstacle_{i}           "
            f"if(x>={xmin:.6f} & x<={xmax:.6f} & y>={ymin:.6f} & y<={ymax:.6f}, 1, 0)\n"
        )
        f.write(
            f"critere_fct_caracteristique fct_car_obstacle_{i}       "
            f"[fct_obstacle_{i}, 0]\n"
        )
        f.write(
            f"cree_entite_avec_critere    obstacle_{i}               "
            f"ENTITE_TOUTES_COMPOSANTES(fct_car_obstacle_{i})\n\n"
        )


print("Obstacle definitions written to Linear_Features.champs")


# ------------------------------------------------------------
# Sous-domaine = domaine entier - tous les obstacles
# ------------------------------------------------------------

with open("Linear_Features.champs", "a") as f:

    f.write("# ------------------------------------------------------------\n")
    f.write("# Sous-domaine sans obstacles\n")
    f.write("# ------------------------------------------------------------\n")

    # If no roads, just write the full domain
    if target_coverage == 0 or len(roads) == 0:
        f.write("cree_entite sous_domaine contour_domaine\n")

    else:
        subtraction = " - ".join([f"obstacle_{i}" for i in range(1, len(roads)+1)])
        f.write(f"cree_entite sous_domaine contour_domaine - {subtraction}\n")



# ------------------------------------------------------------
# Export Neumann boundary conditions for all species
# ------------------------------------------------------------

with open("Linear_Features_BC.champs", "w") as f:

    f.write("# ------------------------------------------------------------\n")
    f.write("# Conditions de Neumann pour chaque obstacle et chaque espèce\n")
    f.write("# ------------------------------------------------------------\n\n")

    species = ["P", "C", "M", "D1"]

    for i in range(1, len(roads)+1):

        f.write(f"# ROAD {i}\n")

        for sp in species:

            # Choose flux term depending on species
            flux_term = "flux_P" if sp == "P" else "flux_prey"

            f.write(
                f'neumann scalaire "bas_obstacle_{i}"    "{sp}" "{flux_term}" "schema_intg"\n'
            )
            f.write(
                f'neumann scalaire "haut_obstacle_{i}"   "{sp}" "{flux_term}" "schema_intg"\n'
            )
            f.write(
                f'neumann scalaire "gauche_obstacle_{i}" "{sp}" "{flux_term}" "schema_intg"\n'
            )
            f.write(
                f'neumann scalaire "droite_obstacle_{i}" "{sp}" "{flux_term}" "schema_intg"\n'
            )

            f.write("\n")

print("Boundary conditions written to Linear_Features_BC.champs")
