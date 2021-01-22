# Regional Urban Growth

These scripts form the model and analyses for the manuscript Pattern-Oriented Calibration and Validation of Urban Growth Models: Case Studies of Dublin, Milan and Warsaw.

The file `parameters.py` contains the model settings, most importantly the case study name ('IE', 'IT', or 'PL', for the Irish, Italian, and Polish case study respectively), step size for the brute force calibration (0.1 in the manuscript) and the time periods to be used for calibration and validation.

The model was adapted for running on PALMA-II High Performance Computing cluster of University of Muenster:
https://confluence.uni-muenster.de/display/HPC
The batch or job scheduling system on PALMA-II is SLURM:
https://slurm.schedmd.com/documentation.html

## Software required:
Python version 3.7, PCRaster version 4.3, matplotlib version 3.3, gdal version 2.4, pandas version 1.2, scipy version 1.6, numba version 0.52, and openpyxl version 3.0.

## Resources required:
Simulation run for the manuscripted required 64GB RAM. Computation time was ~24h for running and calibrating the model for a case study. 

## Running the model
To run the model and analyses for a case study, submit to the cluster the seven numbered jobs in order:
1_parameter_list.sh 
Creates a list of input paramaters for urban growth drivers used in every loop run of the model. This job can be run only once for all case studies tested.

2_LU_urb.sh 
Runs the model in parallel arrays for the given case study. Command #SBATCH --array defines number of arrays, which needs to be adjusted in relation to the step size. 71 arrays were assigned for step size 0.1.

3_check.sh
Checks the completion of array jobs and provides the ID numbers of jobs that need to be re-run

4_re-run.sh
Re-reuns the jobs identified in previous job.

5_transform.sh
Transforms results of the simulation into numpy arrays storing urban / non-urban maps, allocation acuracy metrics and landscape metrics values.

### Following jobs should be submitted after running the simulation for all case studies:
6_visualize.sh
Produce figures visualizing modelling results.

7_compress.sh
Compress the numpy arrays storing simulation results into gzip files.
