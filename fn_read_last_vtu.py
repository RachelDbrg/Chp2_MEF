import glob
import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import gc

def read_last_vtu(iter_dir, type_run):

    if type_run == "para":

        files = sorted(
            glob.glob(os.path.join(iter_dir, "*cont.pvtu"))
        )

        reader = vtk.vtkXMLPUnstructuredGridReader()

    elif type_run == "seq":

        files = sorted(
            glob.glob(os.path.join(iter_dir, "*cont.vtu"))
        )

        reader = vtk.vtkXMLUnstructuredGridReader()

    else:

        raise ValueError(
            f"Unknown type_run = {type_run}"
        )

    if len(files) == 0:

        raise RuntimeError(
            f"No output file found in {iter_dir}"
        )

    # dernier timestep
    output_file = max(
    files,
    key=os.path.getmtime)

    print("Reading:", output_file)

    reader.SetFileName(output_file)
    reader.Update()

    grid = reader.GetOutput()


    ## Etape de nettoyage de cellules redondantes si on a de l'overlap pour les jobs en paralleles
    print(
        "Before cleaning:",
        grid.GetNumberOfPoints(),
        grid.GetNumberOfCells()
    )

    clean = vtk.vtkStaticCleanUnstructuredGrid()
    clean.SetInputData(grid)
    clean.Update()

    grid_clean = clean.GetOutput()

    print(
        "After cleaning:",
        grid_clean.GetNumberOfPoints(),
        grid_clean.GetNumberOfCells()
    )

    points = vtk_to_numpy(
        grid_clean.GetPoints().GetData()
    )

    x = points[:, 0]
    y = points[:, 1]

    idx = np.lexsort((y, x))

    pdata = grid_clean.GetPointData()

    C = vtk_to_numpy(
        pdata.GetArray("C")
    )[idx]

    P = vtk_to_numpy(
        pdata.GetArray("P")
    )[idx]

    M = vtk_to_numpy(
        pdata.GetArray("M")
    )[idx]

    if C.sum() > 0:
        C /= C.sum()

    if P.sum() > 0:
        P /= P.sum()

    if M.sum() > 0:
        M /= M.sum()


    C = np.array(C, copy=True)
    P = np.array(P, copy=True)
    M = np.array(M, copy=True)



    del points
    del pdata
    del idx
    del x
    del y
    del reader
    del clean
    del grid
    del grid_clean
    gc.collect()

    return C, P, M
    
    

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vtk.util.numpy_support import vtk_to_numpy
import vtk
import numpy as np


def read_vtu(output_file, type_run):

    if type_run == "para":

        reader = vtk.vtkXMLPUnstructuredGridReader()

    elif type_run == "seq":

        reader = vtk.vtkXMLUnstructuredGridReader()

    else:

        raise ValueError(
            f"Unknown type_run = {type_run}"
        )

    reader.SetFileName(output_file)
    reader.Update()

    grid = reader.GetOutput()

    points = vtk_to_numpy(
        grid.GetPoints().GetData()
    )

    x = points[:, 0]
    y = points[:, 1]

    idx = np.lexsort((y, x))

    pdata = grid.GetPointData()

    densities = {}

    for field in ["C", "M", "P"]:

        arr = pdata.GetArray(field)

        if arr is None:

            print(
                f"Warning: field '{field}' not found "
                f"in {output_file}"
            )

            densities[field] = None

        else:

            values = vtk_to_numpy(arr)[idx]

            if values.sum() > 0:
                values = values / values.sum()

            densities[field] = values

    return densities
    