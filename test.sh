#!/bin/bash
## *****************************************************
## IMPORTANT: Il faut faire "module load MEFPP" *avant* de lancer sbatch avec ce fichier
##            car on y injecte des paramètres via les modules...
## Notez que deux ## indique un commentaire, alors que #SBATCH est une commande pour l'ordonnanceur.
## *****************************************************

## **NE PAS MODIFIER** ces 2 lignes:
##SBATCH --constraint="*** Chargez le module MEFPP avant de lancer sbatch ***" ## Ne pas modifier: Ça assure qu'on a bien chargé le module *avant* de lancer sbatch...
#SBATCH --exclusive                                                           ## Ne pas modifier: Nous assure de ne pas partager un noeud de calcul!

## *****************************************************
## Section de choses à **MODIFIER** selon votre calcul:
## ****************************************************
#SBATCH --job-name=var_c_100iter     ## Le "nom" de la job... détermine aussi le nom de fichier de sortie
#SBATCH --time=00:07:00 ## Durée max (hh:mm:ss)
#SBATCH --account=def-deteix           ## Numéro de l'allocation de Jean Deteix
#SBATCH --mem=10GB                      ## Demande toute la mémoire des noeuds, sans limite blocante par processus.  Ici, on peut demander des noeuds de 186 Go si on écrit 186G.
#SBATCH --mail-user=rachel.dubourg.1@ulaval.ca       ## La liste des personnes à avertir par courriel
#SBATCH --mail-type=ALL     ## Notifications par courriel pour événements: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=var_c_100iter.out
##SBATCH -o %x-%j.out.txt               ## On redirigera les sorties (stdout/stderr) vers ces fichier (nb: %x désigne le nom de la job et %j le job_id)
#SBATCH -e %x-%j.err.txt
### A ajuster si job roule en //
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --ntasks-per-node=32   # instead of 64 or 128


#echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"


# **CHANGER** Le nb de processus désirés par "noeud" (si 0 alors 1 processus/coeur):
#export nb_processus_par_noeud=$((SLURM_CPUS_ON_NODE / 2))
#export nb_processus_par_noeud=1
#export nb_processus_par_noeud=8

#module load MEFPP 

#module load MEFPP 
#module load scipy-stack/2025a 
#module load python/3.14.2
#module load vtk



#eval "${MEFPP_ENVMPI}"

#module load  StdEnv/2023  vtk
#export nb_processus_par_noeud=$(( SLUR_CPUS_ON_NODE / 2))
export nb_processus_par_noeud=6
#export nb_processus_par_noeud=2

# Option pour "voir" les bindings faits:
# export MEFPP_MPI_REPORTS=GIREF_REPORT


#module load MEFPP

#eval "${MEFPP_ENVMPI}"

#echo "After MEFPP: nb_processus_par_noeud=$nb_processus_par_noeud"

# **Important**: Il faut que la ligne suivante apparaissent après avoir renseigné la variable nb_processus_par_noeud:


module load MEFPP; eval "${MEFPP_ENVMPI}"
module load scipy-stack/2024a 
module load hdf5-mpi/1.14.6
module load petsc/3.25.1
module load python/3.12.4
module load mpi4py/4.1.0
#module load python/3.11.5 
module load vtk
#module hdf5-mpi/1.14.6


# NB: Sur Niagara, ce qu'on appelle ici des coeurs/CPUS deviennent plutôt des hyperthreads (donc au nombre de 80 par noeuds)
#export nb_processus_par_noeud=$(( SLURM_CPUS_ON_NODE / 2))
echo "Before MEFPP: nb_processus_par_noeud=$nb_processus_par_noeud"
# La variable $MPIEXEC_MEFPP contient tous les "bons" paramètres pour lancer MEF++ avec le bon nombre de processus, bindings, etc...
# Le binaire lancé est celui contenu dans variable $MEFPP_BINAIRE qui est MEF++.opt par défaut.
# Pour utiliser mefpp4py, il faut faire:
#   export MEFPP_BINAIRE=python3
#

lscpu | grep Socket
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "SLURM_MEM_PER_NODE = $SLURM_MEM_PER_NODE"



# **Ajouter** les arguments après $MPIEXEC_MEFPP:
# Exemples:
#
#   # Ceci appellera MEF++.opt avec "feracheval" comme argument:
#   $MPIEXEC_MEFPP carreRachel

#   # Ceci appellera python3 avec "monscript.py" et "feracheval" comme arguments:
   export MEFPP_BINAIRE=python3
#   $MPIEXEC_MEFPP main.py 
   #$MPIEXEC_MEFPP figures.py 
   $MPIEXEC_MEFPP Nelder_Mead_parameters_estimation.py
   #$MPIEXEC_MEFPP parms_variations.py

#$MPIEXEC_MEFPP



