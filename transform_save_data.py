# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import metrics
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt

#### Script to read in the metrics saved as the result of the LU_urb.py script.
#### Metrics are transformed into an array

# Get metrics
metricNames = parameters.getSumStats() 

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps=parameters.getNrTimesteps()
numberOfIterations = parameters.getNumberofIterations(parameters.getSuitFactorDict(), parameters.getParametersforCalibration())

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
    
    for i in iterations:
      # Read in the parameters
      pName = 'parameters_iteration_' + str(i) + '.obj'
      pFileName = os.path.join(resultFolder, str(i), pName)
      filehandler = open(pFileName, 'rb') 
      pData = pickle.load(filehandler)
      pArray = np.array(pData, ndmin=1)
          
      # If we are working with the observed data (CLC data):
      if obs:
        name = generateNameT(basename, timestep)
        fileName = os.path.join('observations', country, 'realizations', str(i), name)
        data = metrics.map2Array(fileName, os.path.join('input_data', country, 'sampPoint.col'))
        
      # If we are working with the observed values:  
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

def saveSamplesAndTimestepsAsNumpyArray(basename, iterations, timesteps, obs=False):
  # Convert the output of the model into arrays
  output = openPickledSamplesAndTimestepsAsNumpyArray(basename, iterations, timesteps, obs)
  
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
    
#################################
### SAVE OUTPUTS OF THE MODEL ###
#################################
  
# now save all outputs of the model as one array per variable
# so that we can delete all number folders
 
print("Save modelled and observed metrics: ", metricNames)
print("Parameter values are stored in 3 dimensional array [timestep, iteration, metric value for a given zone]")
print("# timestep: year, \n# iteration: set of parameters used, \n# metric: array of values \
of the selected metric for each zone for each set of parameters seperately")

    
# Save for the modelled and observed metrics:
for aVariable in metricNames:
  saveSamplesAndTimestepsAsNumpyArray(aVariable, iterations, timeSteps)
  saveSamplesAndTimestepsAsNumpyArray(aVariable, obsSampleNumbers,obsTimeSteps, True)  

# Print data description
zonesModelled = np.load(os.path.join("results", country, 'metrics', metricNames[0] + '.npy'))
print('number of time steps: ',len(zonesModelled))
print('number of parameter configurations: ',len(zonesModelled[0]))                     
print('number of zones: ',len(zonesModelled[0][0][1]))
print('Done.')


  
  
  


