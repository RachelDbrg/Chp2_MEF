import numpy as np

def setup_model_dependent_parameters(current_model):
    


    # ==========================================
    # Taille du paysage
    # ==========================================

    landscape_sizes = {
        "SM1": 405.606,
        "SM2": 344.779,
    }

    L = landscape_sizes[current_model]

    # ==========================================
    # Paramètres biologiques (en km)
    # ==========================================

    rayon_perception_C = 4.7
    rayon_perception_M = 1.0
    #rayon_perception_P = 0.2
    rayon_perception_P = 5
    rayon_perception_D = 1.0



    coeff_diffusion_C = 1073.1
    coeff_diffusion_M = 142.35
    coeff_diffusion_P = 1051.25
    coeff_diffusion_D = coeff_diffusion_M

    advection_rate_D = 5

    # ==========================================
    # Paramètres normalisés
    # ==========================================

    rayon_perception_C_norm = rayon_perception_C / L
    rayon_perception_M_norm = rayon_perception_M / L
    rayon_perception_P_norm = rayon_perception_P / L
    rayon_perception_D_norm = rayon_perception_D / L

    rayon_perception_C_norm_Helmotz = (rayon_perception_C_norm/(2*np.sqrt(2)))**2
    rayon_perception_M_norm_Helmotz = (rayon_perception_D_norm/(2*np.sqrt(2)))**2
    rayon_perception_P_norm_Helmotz = (rayon_perception_P_norm/(2*np.sqrt(2)))**2
    rayon_perception_D_norm_Helmotz = (rayon_perception_D_norm/(2*np.sqrt(2)))**2

    coeff_diffusion_C_norm = 1.0

    coeff_diffusion_M_norm = (
        coeff_diffusion_M / coeff_diffusion_C
    )

    coeff_diffusion_P_norm = (
        coeff_diffusion_P / coeff_diffusion_C
    )

    coeff_diffusion_D_norm = (
        coeff_diffusion_D / coeff_diffusion_C
    )

    advection_rate_D_norm = (
        advection_rate_D * L
        / coeff_diffusion_C
    )

    temps_caracteristique = L**2/coeff_diffusion_C

    # ==========================================
    # Listes utilisées dans le GFC
    # ==========================================

    name_parm = [

        "L",

        "rayon_perception_C_norm",
        "rayon_perception_M_norm",
        "rayon_perception_P_norm",
        "rayon_perception_D_norm",

        "coeff_diffusion_C_norm",
        "coeff_diffusion_M_norm",
        "coeff_diffusion_P_norm",
        "coeff_diffusion_D_norm",

        "advection_rate_D_norm",

        "temps_caracteristique",

        "rayon_perception_C_norm_Helmotz",
        "rayon_perception_M_norm_Helmotz",
        "rayon_perception_P_norm_Helmotz",
        "rayon_perception_D_norm_Helmotz"
    ]

    value_parm = [

        L,

        rayon_perception_C_norm,
        rayon_perception_M_norm,
        rayon_perception_P_norm,
        rayon_perception_D_norm,

        coeff_diffusion_C_norm,
        coeff_diffusion_M_norm,
        coeff_diffusion_P_norm,
        coeff_diffusion_D_norm,

        advection_rate_D_norm,

        temps_caracteristique,

        rayon_perception_C_norm_Helmotz,
        rayon_perception_M_norm_Helmotz,
        rayon_perception_P_norm_Helmotz,
        rayon_perception_D_norm_Helmotz
    ]

    # ==========================================
    # Paramètres physiques
    # ==========================================

    physical_parameters = {
        "coeff_diffusion_C": coeff_diffusion_C,
        "coeff_diffusion_M": coeff_diffusion_M,
        "coeff_diffusion_P": coeff_diffusion_P,
        "coeff_diffusion_D": coeff_diffusion_D,
        "advection_rate_D": advection_rate_D,
        "L": L,
    }

    print(rayon_perception_C_norm_Helmotz)

    return (
        name_parm,
        value_parm,
        physical_parameters,
    )