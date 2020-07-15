#!/bin/bash
 
#SBATCH --export=NONE               # Start with a clean environment
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=72        # the number of CPU cores per node
#SBATCH --mem=16G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition=express         # on which partition to submit the job
#SBATCH --time=01:00:00             # the max wallclock time (time limit your job will run)

#SBATCH --job-name=re-run                 # the name of your job
#SBATCH --output=R_%j.dat                 # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=ALL                   # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=k_goch01@uni-muenster.de # your mail address
 
# LOAD MODULES
module load foss/2019a
module load Python/3.7.2
module load PCRaster
module load GDAL/3.0.0-Python-3.7.2

# START THE APPLICATION
echo task: $SLURM_ARRAY_TASK_ID

python /home/k/k_goch01/urban_growth_model/LU_urb.py 76
sleep 5
#python /home/k/k_goch01/urban_growth_model/LU_urb.py 4
#sleep 5
#python /home/k/k_goch01/urban_growth_model/LU_urb.py 28
#sleep 5
#python /home/k/k_goch01/urban_growth_model/LU_urb.py 132
#sleep 5
