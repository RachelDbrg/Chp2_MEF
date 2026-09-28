import glob
import re
import vtk
import pandas as pd
import os
from vtk.util.numpy_support import vtk_to_numpy
import csv
import numpy as np


def progress_bar(current, total, bar_length=40):
    fraction = current / total
    filled = int(fraction * bar_length)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(
        f"\r[{bar}] {fraction*100:5.1f}% ({current}/{total})",
        end=""
    )


def vtu_to_csv(prefix, output_file):

    # Fichier csv de sortie
    output_csv = os.path.join(
        prefix,
        f"{output_file}.csv"
    )

    files = sorted(
        glob.glob(
            os.path.join(prefix, "*.cont.vtu")
        )
    )

    #files = files[:1]

    # Garder 3 snapshots : début, milieu et fin
    nfiles = len(files)

    if nfiles >= 3:
        idx_start = 0
        idx_mid   = nfiles // 2
        idx_end   = nfiles - 1

        files = [
            files[idx_start],
            files[idx_mid],
            files[idx_end]
        ]


    print(f"Found {len(files)} VTU files")

    if len(files) == 0:
        raise RuntimeError(
            f"No .cont.vtu files found in {prefix}"
        )

    # Supprime l'ancien csv
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Existing file deleted → {output_csv}")

    # Création du csv
    with open(output_csv, "w", newline="") as fout:

        writer = csv.writer(fout)

        writer.writerow(
            ["x", "y", "TimeStep", "Field", "Value"]
        )

    max_time = None
    total_files = len(files)

    print("Converting VTU files...")

    for idx, f in enumerate(files, start=1):

        progress_bar(idx, total_files)

        basename = os.path.basename(f)

        # Exemple :
        # NM_SM11818.cont.vtu
        # NM_SM11819.cont.vtu
        match = re.search(
            r"(\d+)(?=\.cont\.vtu$)",
            basename
        )

        if not match:
            print(f"\nSkipping {basename} (no timestep found)")
            continue

        time = int(match.group(1))

        if max_time is None or time > max_time:
            max_time = time

        # Lecture du VTU
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(f)
        reader.Update()

        grid = reader.GetOutput()

        if grid.GetNumberOfPoints() == 0:
            continue

        points = vtk_to_numpy(
            grid.GetPoints().GetData()
        )

        point_data = grid.GetPointData()

        n = points.shape[0]

        with open(output_csv, "a", newline="") as fout:

            writer = csv.writer(fout)

            for i in range(
                point_data.GetNumberOfArrays()
            ):

                name = point_data.GetArrayName(i)

                arr = vtk_to_numpy(
                    point_data.GetArray(i)
                )

                # Champ scalaire
                if arr.ndim == 1:

                    rows = np.column_stack(
                        [
                            points[:, 0],
                            points[:, 1],
                            np.full(n, time),
                            np.full(n, name),
                            arr
                        ]
                    )

                    writer.writerows(rows)

                # Champ vectoriel
                else:

                    for j in range(arr.shape[1]):

                        field_name = f"{name}_{j}"

                        rows = np.column_stack(
                            [
                                points[:, 0],
                                points[:, 1],
                                np.full(n, time),
                                np.full(n, field_name),
                                arr[:, j]
                            ]
                        )

                        writer.writerows(rows)

    print("\n")
    print(f"Maximum TimeStep found → {max_time}")
    print(f"Saved → {output_csv}")

    return output_csv