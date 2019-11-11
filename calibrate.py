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
obsTimeSteps = [10] # for a whole run should be amended # year 2000 and 2006. maybe should go to parameters?

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

def createZeroArray(obsTimeSteps,zonesModelled, obs=False):
  # Get the number of parameter configurations
  noParameterConfigurations = zonesModelled.shape[1]
  # Get the number of zones
  numberZones = len(zonesModelled[0][0][1])
  if obs:
    theArray = np.zeros((len(obsTimeSteps),1,numberZones,1))
  else:
    theArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,numberZones,1))
  return theArray

def storeMetricValues(theArray,obsTimeSteps,zonesModelled, obs=False):
  rowIndex = 0
  if obs:
    for obsTimeStep in range(0,len(obsTimeSteps)):
      theArray[rowIndex,0] = zonesObserved[obsTimeStep,0][1]
      rowIndex = rowIndex+1
  else:
    for obsTimeStep in obsTimeSteps:
      rowIndex = 0
      for p in range(0,zonesModelled.shape[1]): # Loop the parameter configurations
        theArray[rowIndex,p] = zonesModelled[obsTimeStep-1,p][1]
      rowIndex = rowIndex+1

def saveTheArray(output, resultFolder, metricName): 
  # Set the name of the file
  fileName = os.path.join(resultFolder, metricName + '_diff')

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, output)

  print('Difference between modelled and observed values are stored in ', metricName + '_diff.npy')


###########################
### CALIBRATE THE MODEL ###
###########################
 
print("CALIBRATE THE MODEL")

# Calibration of the modell will be based on finding
# minimum root-mean-square error between the metrics modelled and observed.

for aVariable in metricList:
  print('Metric: ',aVariable)
  zonesModelled = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))
  zonesObserved = np.load(os.path.join("results", country, 'metrics', aVariable + '_obs.npy'))
  print('Number of parameter configurations: ',zonesModelled.shape[1])
  
  # 1. Create a list containing possible parameter configurations
  allParameters = getParameterConfigurations(zonesModelled)
  
  # 2. Create the arrays to store the modelled, the observed and the difference metrics values
  modelledArray = createZeroArray(obsTimeSteps,zonesModelled)
  obsArray = createZeroArray(obsTimeSteps,zonesModelled,obs=True)

  # 3. Store the metric values 
  storeMetricValues(modelledArray,obsTimeSteps,zonesModelled)
  storeMetricValues(obsArray,obsTimeSteps,zonesModelled,obs=True)
  
  # 4. Store the difference
  differenceArray = numpy.subtract(modelledArray,obsArray)
  #print(differenceArray)
  
  # 5. Save the data
  saveTheArray(differenceArray, arrayFolder, aVariable)

  


# 1. Calculate Mean Squared Error
numberOfDataPoints = 1 # number of observations. Here: 2000

 
  
  


