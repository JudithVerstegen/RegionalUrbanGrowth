import pickle
import parameters
import os
import gzip
import numpy as np

#### Script to zip al the result npy files
# Got to the directory with results
workDir = parameters.getWorkDir()
cases = parameters.getCaseStudies()
# Loop countries
for country in cases:
  resultFolder = os.path.join(workDir,'results',country, 'metrics')

  # List all the files
  files = os.listdir(resultFolder)
  # Filter files with extension .gz
  filtered_files = [ffile for ffile in files if ffile.endswith(".gz")]
  # Remove filtered files
  for rf in filtered_files:
    r_path = os.path.join(resultFolder,rf)
    os.remove(r_path)
  # Update the list
  files = os.listdir(resultFolder)
  print(files) 
  # zip every file
  for afile in files:
      print("zipping",afile)
      file_path = os.path.join(resultFolder, afile)
      my_array = np.load(file_path,allow_pickle=True)
      f = gzip.GzipFile(os.path.join(resultFolder, afile+".gz"), "w")
      np.save(file=f, arr=my_array)
      f.close()
print('done')
