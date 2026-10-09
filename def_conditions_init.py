import pandas as pd
import os

def setup_initial_conditions(
    current_model,
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

    etalementCloche = 300

    densiteInitC = 1
    densiteInitM = 1
    densiteInitP = 1
    densiteInitD = 0

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

    theoretical_ci = {
        "C": gaussian_sum_C,
        "M": gaussian_sum_M,
        "P": gaussian_sum_P
    }

    # ==================================================
    # Debug CI - gaussienne centrées
    # ==================================================

    debug_ci = {

        "C": (
            "densiteInitC*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),

        "M": (
            "densiteInitM*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),

        "P": (
            "densiteInitP*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),

        "D": (
            "densiteInitD*exp("
            "-etalementCloche*((x-0.5)^2+(y-0.5)^2)"
            ")"
        ),
    }

    # ==================================================
    # Read empirical data
    # ==================================================

    species = species_by_model[current_model]

    data_file = files_by_model[current_model]

    df = pd.read_csv(data_file)
    df = df.sort_values(["x", "y"])

    uds = {}

    for sp in species:

        col = f"{sp}_CF"

        uds[sp] = df[col].values
        uds[sp] /= uds[sp].sum()

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
                f"scalquad C densiteInitC*({theoretical_ci['C']})\n"
            )

            fp.write(
                f"scalquad M densiteInitM*({theoretical_ci['M']})\n"
            )

            fp.write(
                f"scalquad P densiteInitP*({theoretical_ci['P']})\n"
            )

            fp.write(
                "scalaire custom_D (y < 0.02)\n"
            )

            fp.write(
                "scalquad D densiteInitD*custom_D\n"
            )

        elif ci_type == "debug":

            fp.write(
                f"scalquad C densiteInitC*({debug_ci['C']})\n"
            )

            fp.write(
                f"scalquad M densiteInitM*({debug_ci['M']})\n"
            )

            fp.write(
                f"scalquad P densiteInitP*({debug_ci['P']})\n"
            )

            fp.write(
                f"scalquad D densiteInitD*({debug_ci['D']})\n"
            )

    # ==================================================
    # Validation UDs
    # ==================================================

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
        empirical_ci
    )