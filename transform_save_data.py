# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import metrics
import numpy as np
import parameters
import calibrate
from pcraster.framework import *
import matplotlib.pyplot as plt

#### Script to read in the metrics saved as the result of the LU_urb.py script.
#### Metrics are transformed into an array

# Get metrics
metricNames = parameters.getSumStats()
areaMetricNames = ['pd_cal','pd_val','cilp_cal','cilp_val']

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps=parameters.getNrTimesteps()

numberOfIterations = parameters.getNumberofIterations()

iterations = range(1, numberOfIterations+1, 1)

timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsSampleNumbers = [1] #range(1,20+1,1) <- for stochastic model
obsTimeSteps = parameters.getObsTimesteps()

# Path to the folder with the metrics stored
country = parameters.getCountryName()
resultFolder = os.path.join(os.getcwd(),'results',country)
output_mainfolder = os.path.join(resultFolder, "metrics")

#################
### FUNCTIONS ###
#################


def openPickledSamplesAndTimestepsAsNumpyArray(basename,iterations,timesteps, \
                                               obs=False):
  output=[]

  for timestep in timesteps:
    allIterations=[]
    
    for i in iterations: # Loop parameters
      # Read in the parameters
      pName = 'parameters_iteration_' + str(i) + '.obj'
      pFileName = os.path.join(resultFolder, str(i), pName)
      filehandler = open(pFileName, 'rb') 
      pData = pickle.load(filehandler)
      pArray = np.array(pData, ndmin=1)

      # If we are working with urban data, then we are using an array for each cell.
      # Metrics cilp and pd are calculate for the whole area.
      # Other metrics are calculated for each zone:
      if basename == 'urb':
        refArray = 'sampPointNr.col' # Array with an unique ID for each cell
      elif basename in ['cilp','pd']:
        refArray = 'sampSinglePoint.col' # Array with an unique ID for the whole study area
      else:
        refArray = 'sampPoint.col' # Array with an unique ID for each zone
          
      # If we are working with the observed data (CLC data):
      if obs:
        name = generateNameT(basename, timestep)
        fileName = os.path.join('observations', country, 'realizations', str(1), name) # for now only realization == 1
        data = metrics.map2Array(fileName, os.path.join('input_data', country, refArray))
        
      # If we are working with the modelled values:  
      else:
        theName = basename + str(timestep) + '.obj'
        fileName = os.path.join(resultFolder, str(i), theName)
        filehandler = open(fileName, 'rb')
        data = pickle.load(filehandler)
        
        '''# Keep these lines for the later use:
        if it is a dictionary, get the sugar cane parameters (lu type 6)
        if type(data) == dict:
          data = data.get(1)
        filehandler.close()'''

        
      # if the loaded data was not yet an array, make it into one
      # minimum number of dimensions is one, to prevent a zero-dimension array
      # (not an array at all)
      array = np.array(data, ndmin=1)
      # add an extra dimension that would normally be y, if the data was a map
      # so that Derek's plot functions can be used
      array = array.reshape(len(array),1)
      allIterations.append([pArray,array])
    output.append(allIterations)
  outputAsArray = np.array(output)

  return outputAsArray

def setNameClearSave(basename, output, obs=False):
  # Check if the directory exists. If not, create.
  if not os.path.isdir(output_mainfolder):
      os.mkdir(output_mainfolder)
      
  # Set the name of the file
  if obs:
    fileName = os.path.join(output_mainfolder, basename + '_obs')
  else:
    fileName = os.path.join(output_mainfolder, basename)

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, output)
"""    
#################################
### SAVE OUTPUTS OF THE MODEL ###
#################################

########### Save the observed metrics and urban areas
# Metrics:
for aVariable in metricNames + areaMetricNames:  
  output_obs = openPickledSamplesAndTimestepsAsNumpyArray(aVariable, obsSampleNumbers,obsTimeSteps, True)
  setNameClearSave(aVariable, output_obs,obs=True)

# Urban areas
output_urb_obs = openPickledSamplesAndTimestepsAsNumpyArray('urb', obsSampleNumbers,obsTimeSteps, True)
setNameClearSave('urb', output_urb_obs,obs=True)

########### Save the modelled metrics and urban areas
# Metrics:
for aVariable in metricNames + areaMetricNames:
  output_mod = openPickledSamplesAndTimestepsAsNumpyArray(aVariable, iterations, timeSteps, False)
  setNameClearSave(aVariable, output_mod,obs=False)

# Modellled urban areas only for the observation years:
for a_step in obsTimeSteps:
  subset_urb_mod = openPickledSamplesAndTimestepsAsNumpyArray('urb', iterations, [a_step], False)
  setNameClearSave('urb_subset_'+str(a_step), subset_urb_mod,obs=False)

# Parameter sets
parameter_sets = subset_urb_mod[0,:,0]
setNameClearSave('parameter_sets', parameter_sets, obs=False)

print("Modelled and observed metrics and urban areas saved as npy files")

'''
########### Delete all number folders
files = os.listdir(resultFolder)
for f in files:
    if f not in ['metrics']:
      shutil.rmtree(os.path.join(resultFolder, f))
print("All number folders deleted.")
'''

##################################
### CALCULATE KAPPA STATISTICS ###
##################################

########### Calculate Kappa statistics
# Calculate Kappa standard and Kappa statistic for the whole study area
calibrate.calculateKappa()
calibrate.calculateKappaSimulation()"""

# Calculate Kappa standard and Kappa statistic for the calibration and validation based on selected zones
for scenario in [1,2]:
  for aim in ['calibration','validation']:
    #calibrate.calculateKappa(3,aim)
    calibrate.calculateKappaSimulation(scenario,aim)
print("Kappa statistics calculated and saved as npy file")

##############################
### CALIBRATE AND VALIDATE ###
##############################

# Get the calibration and validation results for 3 scenarios:
# 1: calibrate on year 2000-2006 valdate on 2012-2018
# 2: calibrate on years 2012-2018 validate on 200-2006
# 3: calibrate and validate on different aras (zones)
"""
for scenario in [1,2,3]:
  print('Scenario '+str(scenario))
  # Calibrate, validate and save the results as csv file
  calibrate.calibrate_validate(scenario)
  print("Model calibrated and validated")
  # Perform validation based on multi-objective calibration
  calibrate.multiobjective(scenario) 
  print("Model calibrated using multi-objective goal function and validated")"""

  


