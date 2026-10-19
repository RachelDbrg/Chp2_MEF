def apply_initial_conditions(
    gfc,
    ci_description
):
    """
    Réassigne explicitement les conditions initiales
    aux champs biologiques.
    """

    # paramètres des CI
    gfc.reqChamp("etalementCloche").asgnValeur(
        ci_description["etalementCloche"]
    )

    gfc.reqChamp("densiteInitC").asgnValeur(
        ci_description["densiteInitC"]
    )

    gfc.reqChamp("densiteInitM").asgnValeur(
        ci_description["densiteInitM"]
    )

    gfc.reqChamp("densiteInitP").asgnValeur(
        ci_description["densiteInitP"]
    )

    gfc.reqChamp("densiteInitD").asgnValeur(
        ci_description["densiteInitD"]
    )

    print("Conditions initiales réassignées")

    print({
        "etalementCloche": ci_description["etalementCloche"],
        "densiteInitC": ci_description["densiteInitC"],
        "densiteInitM": ci_description["densiteInitM"],
        "densiteInitP": ci_description["densiteInitP"],
        "densiteInitD": ci_description["densiteInitD"],
    })


    