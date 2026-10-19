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



def vtu_to_csv(
    prefix,
    output_file,
    n_snapshots=None
):

    output_csv = os.path.join(
        prefix,
        f"{output_file}.csv"
    )

    files = sorted(
    glob.glob(
        os.path.join(
            prefix,
            "snapshot_*.cont.vtu"
        )
    ),
    key=lambda x: int(
        re.search(
            r"snapshot_(\d+)\.cont\.vtu$",
            os.path.basename(x)
        ).group(1)
    )
)



    if len(files) == 0:

        raise RuntimeError(
            f"No .cont.vtu files found in {prefix}"
        )

    # ------------------------------------------------------------------
    # Select snapshots
    # ------------------------------------------------------------------

    if n_snapshots is not None:

        n_snapshots = min(
            n_snapshots,
            len(files)
        )

        idx = np.linspace(
            0,
            len(files) - 1,
            n_snapshots
        ).round().astype(int)

        idx = np.unique(idx)

        files = [
            files[i]
            for i in idx
        ]

    print(
        f"Found {len(files)} VTU files"
    )

    # ------------------------------------------------------------------
    # Delete existing CSV
    # ------------------------------------------------------------------

    if os.path.exists(output_csv):

        os.remove(output_csv)

        print(
            f"Existing file deleted → "
            f"{output_csv}"
        )

    # ------------------------------------------------------------------
    # Create CSV header
    # ------------------------------------------------------------------

    with open(
        output_csv,
        "w",
        newline=""
    ) as fout:

        writer = csv.writer(fout)

        writer.writerow(
            [
                "x",
                "y",
                "TimeStep",
                "Field",
                "Value"
            ]
        )

    # ------------------------------------------------------------------
    # Convert files
    # ------------------------------------------------------------------

    max_time = None

    total_files = len(files)

    print(
        "Converting VTU files..."
    )

    for i_file, f in enumerate(
        files,
        start=1
    ):

        progress_bar(
            i_file,
            total_files
        )

        basename = os.path.basename(f)


        # match = re.search(
        #     r"(\d+)\.cont\.vtu$",
        #     basename
        # )

        match = re.search(
            r"snapshot_(\d+)\.cont\.vtu$",
            basename
        )
        

        if match is None:

            print(
                f"\nSkipping "
                f"{basename}"
            )

            continue

        time = int(
            match.group(1)
        )

        if (
            max_time is None
            or time > max_time
        ):
            max_time = time

        # --------------------------------------------------------------
        # Read VTU
        # --------------------------------------------------------------

        reader = (
            vtk.vtkXMLUnstructuredGridReader()
        )

        reader.SetFileName(f)

        reader.Update()

        grid = reader.GetOutput()

        if (
            grid.GetNumberOfPoints()
            == 0
        ):
            continue

        points = vtk_to_numpy(
            grid.GetPoints().GetData()
        )

        point_data = (
            grid.GetPointData()
        )

        n = points.shape[0]

        with open(
            output_csv,
            "a",
            newline=""
        ) as fout:

            writer = csv.writer(
                fout
            )

            for j in range(
                point_data.GetNumberOfArrays()
            ):

                name = (
                    point_data.GetArrayName(j)
                )

                arr = vtk_to_numpy(
                    point_data.GetArray(j)
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

                    for k in range(
                        arr.shape[1]
                    ):

                        field_name = (
                            f"{name}_{k}"
                        )

                        rows = np.column_stack(
                            [
                                points[:, 0],
                                points[:, 1],
                                np.full(n, time),
                                np.full(n, field_name),
                                arr[:, k]
                            ]
                        )

                        writer.writerows(rows)

    print("\n")

    print(
        f"Maximum TimeStep found → "
        f"{max_time}"
    )

    print(
        f"Saved → {output_csv}"
    )

    return output_csv