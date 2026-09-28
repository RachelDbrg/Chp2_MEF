import os
import glob
import re


def get_scenario_vtus(source_dir):

    scenario_vtus = {}

    iter_dirs = sorted([
        os.path.join(source_dir, d)
        for d in os.listdir(source_dir)
        if d.startswith("iter_")
        and os.path.isdir(os.path.join(source_dir, d))
    ])

    for iter_dir in iter_dirs:

        files = glob.glob(
            os.path.join(iter_dir, "*.cont.vtu")
        )

        files.sort(
            key=lambda f: int(
                re.search(
                    r"_(\d+)\.cont\.vtu$",
                    os.path.basename(f)
                ).group(1)
            )
        )

        scenario_vtus[
            os.path.basename(iter_dir)
        ] = files

    return scenario_vtus