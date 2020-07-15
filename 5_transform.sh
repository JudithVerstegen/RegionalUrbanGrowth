#!/bin/bash
 
#SBATCH --export=NONE               # Start with a clean environment
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=72        # the number of CPU cores per node
#SBATCH --mem=64G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition=express         # on which partition to submit the job
#SBATCH --time=01:00:00             # the max wallclock time (time limit your job will run)

#SBATCH --job-name=transform              # the name of your job
#SBATCH --output=R_%A.dat              # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=ALL                   # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=k_goch01@uni-muenster.de # your mail address
 
# LOAD MODULES
module load foss/2019a
module load Python/3.7.2
module load PCRaster
#module load GDAL/3.0.0-Python-3.7.2
module load numba 

# START THE APPLICATION
python /home/k/k_goch01/urban_growth_model/transform_save_data.py
