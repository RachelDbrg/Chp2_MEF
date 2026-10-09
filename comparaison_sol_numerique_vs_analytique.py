from fn_read_last_vtu import read_vtu_validation
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt




# ============================================================
# Solution analytique de diffusion pure
# ============================================================

def analytical_gaussian(
    x,
    y,
    t,
    amplitude,
    etalement,
    D=1.0,
    x0=0.5,
    y0=0.5
):

    sigma0_sq = 1/(2*etalement)

    sigma_t_sq = sigma0_sq + 2*D*t

    r2 = (x-x0)**2 + (y-y0)**2

    return (
        amplitude
        * sigma0_sq/sigma_t_sq
        * np.exp(
            -r2/(2*sigma_t_sq)
        )
    )


# ============================================================
# Paramètres des espèces
# ============================================================

etalementCloche = 300

densiteInitC = 1
densiteInitM = 1
densiteInitP = 1
densiteInitD = 1 



species_params = {

    "C": {
        "amplitude": densiteInitC,
        "etalement": etalementCloche,
        "D": 1.0
    },

    "M": {
        "amplitude": densiteInitM,
        "etalement": etalementCloche,
        "D": 142.35/1073.1
    },

    "P": {
        "amplitude": densiteInitP,
        "etalement": etalementCloche,
        "D": 1051.25/1073.1
    },

    "D": {
        "amplitude": densiteInitD,
        "etalement": etalementCloche,
        "D": 142.35/1073.1
    }

}



# ============================================================
# Temps
# ============================================================

delta_t = 0.0001

results = []


# ============================================================
# Extraction du numéro de pas
# ============================================================

def extract_step(filename):

    basename = os.path.basename(filename)

    number = (
        basename
        .replace("NM_SM1_", "")
        .replace(".cont.vtu", "")
    )

    return int(
        number[:len(number)//2]
    )


# ============================================================
# Fichiers VTU
# ============================================================

files = sorted(
    glob.glob(
        "simulations/Nelder_Mead/NM_SM1/iter_0000/*.cont.vtu"
    ),
    key=extract_step
)


# ============================================================
# Comparaison numérique vs analytique
# ============================================================

for output_file in files:

    step = extract_step(output_file)

    t = step * delta_t

    data = read_vtu_validation(
        output_file,
        "seq"
    )

    x = data["x"]
    y = data["y"]

    for species, params in species_params.items():

        U_num = data["fields"][species]

        U_exact = analytical_gaussian(
            x=x,
            y=y,
            t=t,
            amplitude=params["amplitude"],
            etalement=params["etalement"],
            D=params["D"]
        )

        err = U_num - U_exact

        L1 = np.mean(np.abs(err))

        L2 = np.sqrt(
            np.mean(err**2)
        )

        Linf = np.max(
            np.abs(err)
        )

        print(
            f"{species} "
            f"step={step:04d} "
            f"t={t:.6f} "
            f"L1={L1:.4e} "
            f"L2={L2:.4e} "
            f"Linf={Linf:.4e}"
        )

        results.append(
            [
                species,
                step,
                t,
                L1,
                L2,
                Linf
            ]
        )


# ============================================================
# Sauvegarde CSV
# ============================================================

df = pd.DataFrame(
    results,
    columns=[
        "species",
        "step",
        "time",
        "L1",
        "L2",
        "Linf"
    ]
)

print(df)

df.to_csv(
    "verification_diffusion_4species.csv",
    index=False
)


# ============================================================
# Graphiques
# ============================================================

plt.figure(figsize=(8,5))

for species in species_params:

    df_sp = df[df["species"] == species]

    plt.semilogy(
        df_sp["time"],
        df_sp["L2"],
        label=species
    )

plt.xlabel("Temps")
plt.ylabel("Erreur L2")
plt.legend()
plt.tight_layout()
plt.savefig(
    "comparaison_L2_4_especes.png",
    dpi=300
)