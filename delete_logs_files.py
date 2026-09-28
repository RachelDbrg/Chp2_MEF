import os
import glob

# Motifs des fichiers à supprimer dans le dossier courant
patterns = ["*err.txt", "*_chrono.txt"]

for pattern in patterns:
    for f in glob.glob(pattern):
        print("Suppression :", f)
        os.remove(f)
