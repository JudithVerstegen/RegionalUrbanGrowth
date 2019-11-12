''' Calibration phase of the LU_urb.py'''

import pickle
import os
import metrics
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt

#### Script to find calibrate LU model

# Get metrics
metricList = parameters.getSumStats() 

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps=parameters.getNrTimesteps()
numberOfIterations = parameters.getNumberofIterations(parameters.getSuitFactorDict(), parameters.getParametersforCalibration())

iterations = range(1, numberOfIterations+1, 1)
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsSampleNumbers = [1] #range(1,20+1,1) <- for stochastic model
obsTimeSteps = parameters.getObsTimesteps() # [1,11,17]

# Path to the folder with the metrics npy arrays stored
country = parameters.getCountryName()
arrayFolder = os.path.join(os.getcwd(),'results',country,'metrics')


#################
### FUNCTIONS ###
#################

def getParameterConfigurations(metricsArray):
  allParameters = []
  for i in range(0,len(metricsArray[0])):
    allParameters.append(metricsArray[0][i][0])
  return allParameters

def createDiffArray(modelled,observed):
  # Get the number of parameter configurations
  noParameterConfigurations = modelled.shape[1]
  # Get the number of zones
  numberZones = len(modelled[0][0][1])
  # Create an array: no of rows = no of observed timesteps, no of columns = no of parameter sets
  # Each cell store the values for the zones
  theArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,numberZones,1))
  
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,noParameterConfigurations):
      for zone in range(0,numberZones):
        theArray[row,col][zone][0] = modelled[obsTimeSteps[row]-1,col][1][zone] - observed[row,0][1][zone]  
  return theArray
  
def saveTheArray(output, resultFolder, metricName): 
  # Set the name of the file
  fileName = os.path.join(resultFolder, metricName + '_diff')

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, output)

  print('Difference between modelled and observed values are stored in', metricName + '_diff.npy')


###########################
### CALIBRATE THE MODEL ###
###########################
 
print("CALIBRATE THE MODEL")
print('Observation time steps:',obsTimeSteps)


# Calibration of the modell will be based on finding
# minimum root-mean-square error between the metrics modelled and observed.

for aVariable in metricList:
  zonesModelled = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))
  zonesObserved = np.load(os.path.join("results", country, 'metrics', aVariable + '_obs.npy'))

  print('.')
  print('Metric: ',aVariable)

  # Create the arrays to store the difference between the modelled and the observed and metrics values
  dArray = createDiffArray(zonesModelled,zonesObserved)
  # Save the data
  saveTheArray(dArray, arrayFolder, aVariable)

# 1. Calculate Mean Squared Error
numberOfDataPoints = 1 # number of observations. Here: 2000

 
  
  


