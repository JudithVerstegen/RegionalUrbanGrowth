import os
import parameters
#### Script to check results of the LU model

# Work directory:
work_dir = os.getcwd()#'/scratch/tmp/k_goch01'
country = parameters.getCountryName()
resultFolder = os.path.join(work_dir,'results',country)
# List the folders
files = os.listdir(resultFolder)
# Loop folders
for f in files:
  aFolder = os.path.join(resultFolder,str(f))
  # Check if the metric file is created
  urbFile = os.path.join(aFolder,'cilp1.obj')
  # If there is no urb file, print the INDEX of the parameter set to be redone
  if not os.path.isfile(urbFile):
    print(str(int(f)-1))
