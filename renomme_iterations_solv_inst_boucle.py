import os
import glob
import re


## Definit le dossier avec les simulations
source_dir = "/users/rdubourg/Documents/Chp2_MEF/simulations/var_parms/i/Null_NMP_"

# def get_scenario_vtus(source_dir):

#     scenario_vtus = {}

#     iter_dirs = sorted([
#         os.path.join(source_dir, d)
#         for d in os.listdir(source_dir)
#         if d.startswith("iter_")
#         and os.path.isdir(os.path.join(source_dir, d))
#     ])

#     for iter_dir in iter_dirs:

#         files = glob.glob(
#             os.path.join(iter_dir, "*.cont.vtu")
#         )

#         files.sort(
#             key=lambda f: int(
#                 re.search(
#                     r"_(\d+)\.cont\.vtu$",
#                     os.path.basename(f)
#                 ).group(1)
#             )
#         )

#         scenario_vtus[
#             os.path.basename(iter_dir)
#         ] = files

#     return scenario_vtus


for scenario_dir in glob.glob(os.path.join(source_dir, "scenario_*")):

    files = glob.glob(
        os.path.join(scenario_dir, "*.cont.vtu")
    )

    files.sort(
        key=lambda f: int(
            re.search(
                r"_(\d+)\.cont\.vtu$",
                os.path.basename(f)
            ).group(1)
        )
    )

    # renommage temporaire
    temp_files = []

    for i, old_file in enumerate(files):
        temp_file = os.path.join(
            scenario_dir,
            f"tmp_{i:04d}.vtu"
        )
        os.rename(old_file, temp_file)
        temp_files.append(temp_file)

    # renommage final
    for i, temp_file in enumerate(temp_files):
        final_file = os.path.join(
            scenario_dir,
            f"Null_N_{i:04d}.cont.vtu"
        )
        os.rename(temp_file, final_file)