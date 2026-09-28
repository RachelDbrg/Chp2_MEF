import pandas as pd
import numpy as np
import os

def func_equilibrium_time(data_csv, output_dir, epsilon=1e-3):
    """
    Estimate time to spatial equilibrium for multiple species (C, M, D1, P)
    from PDE output CSV.

    Parameters
    ----------
    data_csv : str
        Path to the CSV file containing PDE outputs.
    output_dir : str
        Directory where the equilibrium results will be saved.
    epsilon : float
        Threshold for equilibrium detection (default = 1e-3).

    Returns
    -------
    results : dict
        Dictionary mapping each species to its equilibrium time step
        (or None if not reached).
    """

    # Load data
    df = pd.read_csv(data_csv)

    # Species columns to evaluate
    species_list = ["C", "M", "D1", "P"]

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for species in species_list:

        # Compute L2 norm of the solution at each time step
        norms = (
            df.groupby("TimeStep")[species]
              .apply(lambda v: np.linalg.norm(v.values))
              .sort_index()
        )

        # Compute relative change between successive time steps
        rel_change = norms.diff().abs() / norms.shift(1)

        # Detect equilibrium
        equilibrium_step = None
        for t in rel_change.index[1:]:  # skip first (no previous step)
            if rel_change.loc[t] < epsilon:
                # Check if all later steps also satisfy the threshold
                if (rel_change.loc[t:] < epsilon).all():
                    equilibrium_step = t
                    break

        # Save result for this species
        results[species] = equilibrium_step

        # Write output file
        out_path = os.path.join(output_dir, f"equilibrium_time_{species}.txt")
        with open(out_path, "w") as f:
            if equilibrium_step is not None:
                f.write(f"Equilibrium reached at time step: {equilibrium_step}\n")
            else:
                f.write("Equilibrium not reached within available time steps.\n")

    return results
