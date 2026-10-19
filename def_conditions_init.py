import pandas as pd
import os

def setup_initial_conditions(
    ci_type,
    species_by_model,
    files_by_model,
    seed=69
):
    """
    Configure les conditions initiales et écrit
    initial_conditions.champs.

    Returns
    -------
    species : list
    df : DataFrame
    uds : dict
    """

    import random

    # ==================================================
    # Empirical CI
    # ==================================================

    empirical_ci = {

        "SM1": {
            "C": "carreRachel.caribou_CI.cmefpp",
            "P": "carreRachel.wolf_CI.cmefpp",
            "M": 0,
            "D": 0,
        },

        "SM2": {
            "C": "carreRachel.caribou_CI.cmefpp",
            "P": 0,
            "M": "carreRachel.moose_CI.cmefpp",
            "D": 0,
        }
    }

    # ==================================================
    # Theoretical CI - 10 gaussiennes aléatoires 
    # ==================================================

    random.seed(seed)

    positions_C = {}
    positions_M = {}
    positions_P = {}
    positions_D = {}

    etalementCloche = 3000

    densiteInitC = 1
    densiteInitM = 1
    densiteInitP = 1
    densiteInitD = 1

    initial_densities = {
    "caribou": densiteInitC,
    "moose":   densiteInitM,
    "wolf":    densiteInitP,
    "deer":    densiteInitD,
}

    active_species = [
    sp for sp, density in initial_densities.items()
    if density > 0
]

    for i in range(10):

        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        positions_C[f"C_{i+1}"] = f"((x-{x})^2 + (y-{y})^2)"

        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        positions_M[f"M_{i+1}"] = f"((x-{x})^2 + (y-{y})^2)"

        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        positions_P[f"P_{i+1}"] = f"((x-{x})^2 + (y-{y})^2)"

    gaussian_sum_C = " + ".join(
        [f"exp((-etalementCloche*C_{i+1}))" for i in range(10)]
    )

    gaussian_sum_M = " + ".join(
        [f"exp((-etalementCloche*M_{i+1}))" for i in range(10)]
    )

    gaussian_sum_P = " + ".join(
        [f"exp((-etalementCloche*P_{i+1}))" for i in range(10)]
    )

    gaussian_sum_D = " + ".join(
            [f"exp((-etalementCloche*D_{i+1}))" for i in range(10)]
        )


    theoretical_ci = {
        "C0": gaussian_sum_C,
        "M0": gaussian_sum_M,
        "P0": gaussian_sum_P,
        "D0": gaussian_sum_D
    }

    # ==================================================
    # Debug CI - gaussienne centrées
    # ==================================================

    debug_ci = {

        "C0": (
            "densiteInitC*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),

        "M0": (
            "densiteInitM*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),

        "P0": (
            "densiteInitP*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),

        "D0": (
            "densiteInitD*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),
    }



    # =================================================
    # Creation d'un dictionnaire avec les ci
    # =================================================

    ci_description = {
    "ci_type": ci_type,
    "etalementCloche": etalementCloche,
    "densiteInitC": densiteInitC,
    "densiteInitM": densiteInitM,
    "densiteInitP": densiteInitP,
    "densiteInitD": densiteInitD,
    "debug_ci": debug_ci,
    "theoretical_ci": theoretical_ci,
}


    positions = {
    "caribou": positions_C,
    "moose": positions_M,
    "wolf": positions_P,
    "deer": positions_D,
}

    # ==================================================
    # Read empirical data
    # ==================================================

    uds = {}
    df = None

    if ci_type == "empirical":

        raise ValueError(
            "Le mode empirical nécessite encore un current_model "
            "(SM1 ou SM2)."
        )

    else:

        species = [
            sp
            for sp, density in initial_densities.items()
            if density > 0
        ]

    # ==================================================
    # Write .champs file
    # ==================================================

    CI_FILE = os.path.join(
        os.path.dirname(__file__),
        "initial_conditions.champs"
    )

    with open(CI_FILE, "w") as fp:

        if ci_type == "empirical":

            for field_name, value in \
                empirical_ci[current_model].items():

                if isinstance(value, str):

                    fp.write(
                        f'scallin {field_name} "{value}"\n'
                    )

                else:

                    fp.write(
                        f"scallin {field_name} {value}\n"
                    )

        elif ci_type in ["theoretical", "debug"]:

            fp.write(
                f"scalaire etalementCloche "
                f"{etalementCloche}\n"
            )

            fp.write(
                f"scalaire densiteInitC {densiteInitC}\n"
            )

            fp.write(
                f"scalaire densiteInitM {densiteInitM}\n"
            )

            fp.write(
                f"scalaire densiteInitP {densiteInitP}\n"
            )

            fp.write(
                f"scalaire densiteInitD {densiteInitD}\n"
            )

        if ci_type == "theoretical":

            for name, expr in positions_C.items():
                fp.write(f"scalaire {name} {expr}\n")

            for name, expr in positions_M.items():
                fp.write(f"scalaire {name} {expr}\n")

            for name, expr in positions_P.items():
                fp.write(f"scalaire {name} {expr}\n")

            fp.write(
                f"scalquad C0 densiteInitC*({theoretical_ci['C0']})\n"
            )

            fp.write(
                f"scalquad M0 densiteInitM*({theoretical_ci['M0']})\n"
            )

            fp.write(
                f"scalquad P0 densiteInitP*({theoretical_ci['P0']})\n"
            )

            # fp.write(
            #     "scalaire custom_D (y < 0.02)\n"
            # )

            # fp.write(
            #     "scalquad D0 densiteInitD*custom_D\n"
            # )

            fp.write(
            "scalaire sigma_D 0.02\n" 
            )
            fp.write(
            "scalquad D0 densiteInitD*exp(-((y-0.0)^2)/(2*sigma_D^2))\n"
            )


        elif ci_type == "debug":

            fp.write(
                f"scalquad C0 densiteInitC*({debug_ci['C0']})\n"
            )

            fp.write(
                f"scalquad M0 densiteInitM*({debug_ci['M0']})\n"
            )

            fp.write(
                f"scalquad P0 densiteInitP*({debug_ci['P0']})\n"
            )

            fp.write(
                f"scalquad D0 densiteInitD*({debug_ci['D0']})\n"
            )

    # ==================================================
    # Validation UDs
    # ==================================================

    if df is not None:

        for sp in species:

            col = f"{sp}_CF"

            values = df[col].values

            print(
                f"{sp}: "
                f"min={values.min():.6e}, "
                f"max={values.max():.6e}, "
                f"sum={values.sum():.12f}"
            )





    return (
    species,
    df,
    uds,
    initial_densities,
    positions,
    ci_description)