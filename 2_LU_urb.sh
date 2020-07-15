#!/bin/bash
 
#SBATCH --export=NONE               # Start with a clean environment
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=72        # the number of CPU cores per node
#SBATCH --mem=16G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition=express         # on which partition to submit the job
#SBATCH --time=01:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --array=0-71                # set the input elements: start, end and step:q

#SBATCH --job-name=LU_urb                 # the name of your job
#SBATCH --output=R_%A-%a.dat              # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=ALL                   # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=k_goch01@uni-muenster.de # your mail address
 
# LOAD MODULES
module load foss/2019a
module load Python/3.7.2
module load PCRaster
module load GDAL/3.0.0-Python-3.7.2

# START THE APPLICATION
echo task: $SLURM_ARRAY_TASK_ID

python /home/k/k_goch01/urban_growth_model/LU_urb.py $(( $SLURM_ARRAY_TASK_ID*4+0 ))
sleep 5
python /home/k/k_goch01/urban_growth_model/LU_urb.py $(( $SLURM_ARRAY_TASK_ID*4+1 ))
sleep 5
python /home/k/k_goch01/urban_growth_model/LU_urb.py $(( $SLURM_ARRAY_TASK_ID*4+2 ))
sleep 5
python /home/k/k_goch01/urban_growth_model/LU_urb.py $(( $SLURM_ARRAY_TASK_ID*4+3 ))
sleep 5
