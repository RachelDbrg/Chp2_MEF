## This script calls the functions that convert the output of the PDEs into .csv, and produce the figures

### Load all the packages
import os
import glob
import re
import vtk
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
from mefpp4py import mefpp
from mpi4py import MPI


### Load the config files, specifying the path and simulations parameters
from config import current_simulation, simulation_folder, figures_folder


### Make sure no simulations already exist for the current config. If so, erase them then run the
### computation (avoid conflicts while producing figures)


#pattern = os.path.join(simulation_folder, f"{current_simulation}*")

#print("pattern=", pattern)

#files = glob.glob(pattern)

#if not files:
#    print("No files to delete.")
#else:
#    for f in files:
#        print("Deleting:", f)
#        os.remove(f)



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Only rank 0 performs deletion
if rank == 0:
    pattern = os.path.join(simulation_folder, f"{current_simulation}*")
    print("pattern=", pattern)

    files = glob.glob(pattern)

    if not files:
        print("No files to delete.")
    else:
        for f in files:
            print("Deleting:", f)
            os.remove(f)

# Synchronize all ranks so they wait until deletion is done
comm.Barrier()



## Run the MEFPP program
#prefix = "carreRachel"
#mefpp.initialise(prefix)
#print("Initilisation du pb")

#mefpp.litEtExecuteActionsDansCollection()


### This is a problem when ran in //
# Only rank 0 should delete files
#if rank == 0:
#    pattern = os.path.join(simulation_folder, f"{current_simulation}*")
#    print("pattern=", pattern)

    #files = glob.glob(pattern)

    #if not files:
    #    print("No files to delete.")
    #else:
    #    for f in files:
    #        print("Deleting:", f)
    #        os.remove(f)



# Initialise MPI
mefpp.initialise("carreRachel")
print("Initialisation du pb")
mefpp.litEtExecuteActionsDansCollection()


