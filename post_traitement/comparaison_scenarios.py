import os
import re
import glob
import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vtk.util.numpy_support import vtk_to_numpy


# --------------------------------------------------
# Lecture d'une simulation
# --------------------------------------------------

def load_simulation(sim_dir, field_name):
    files = glob.glob(
    os.path.join(sim_dir, "*.cont.vtu")
)

    print(f"Found {len(files)} files in {sim_dir}")

    files.sort(
        key=lambda f: int(
            re.search(
                r"(\d+)(?=\.cont\.vtu$)",
                os.path.basename(f)
            ).group(1)
        )
    )

    data = {}

    for f in files:

        basename = os.path.basename(f)

        match = re.search(
            r"(\d+)(?=\.cont\.vtu$)",
            basename
        )

        if match is None:
            continue

        timestep = int(match.group(1))

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(f)
        reader.Update()

        grid = reader.GetOutput()

        point_data = grid.GetPointData()

        arr = point_data.GetArray(field_name)

        if arr is None:
            raise ValueError(
                f"{field_name} not found in {basename}"
            )

        values = vtk_to_numpy(arr)

        values = np.maximum(values, 0)

        if values.sum() > 0:
            values = values / values.sum()

        data[timestep] = values

    return data


# --------------------------------------------------
# Distances
# --------------------------------------------------

def l1_distance(p, q):

    return np.mean(np.abs(p - q))


def bhattacharyya_distance(p, q):

    p = p / p.sum()
    q = q / q.sum()

    bc = np.sum(np.sqrt(p * q))

    bc = np.clip(bc, 1e-15, 1)

    return -np.log(bc)


# --------------------------------------------------
# Comparaison de deux simulations
# --------------------------------------------------

def compare_simulations(
        sim1_dir,
        sim2_dir,
        field_name
):

    sim1 = load_simulation(sim1_dir, field_name)
    sim2 = load_simulation(sim2_dir, field_name)

    common_times = sorted(
        set(sim1.keys()).intersection(
            sim2.keys()
        )
    )

    results = []

    for t in common_times:

        u1 = sim1[t]
        u2 = sim2[t]

        results.append({
            "time": t,
            "L1": l1_distance(u1, u2),
            "BC_distance": bhattacharyya_distance(u1, u2)
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Exemple
# --------------------------------------------------

sim_theta01 = (
    "/home/rdubourg/scratch/Chp2/Nelder_Mead_sep_2026/simulations/theta_01"
)

sim_theta1 = (
    "/home/rdubourg/scratch/Chp2/Nelder_Mead_sep_2026/simulations/theta_1"
)

df = compare_simulations(
    sim_theta01,
    sim_theta1,
    field_name="C"
)

print(df.head())


plt.figure(figsize=(8,5))

plt.plot(
    df["time"],
    df["BC_distance"],
    lw=2,
    alpha = 0.3
)

plt.xlabel("Time step")
plt.ylabel("Bhattacharyya distance")
plt.title("theta=0.1 vs theta=1")

plt.tight_layout()

plt.savefig(
    "theta01_vs_theta1.png",
    dpi=300
)

print("Figure saved")