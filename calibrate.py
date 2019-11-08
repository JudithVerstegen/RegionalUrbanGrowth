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
#metricList = parameters.getSumStats() 
metricList = ['np'] # FOR TESTING

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

# Create a list of parametr configurations
def getParameterConfigurations(metricsArray):
  allParameters = []
  for i in range(0,len(metricsArray[0])):
    allParameters.append(metricsArray[0][i][0])
  print('Number of parameter configurations: ',metricsArray.shape[1])
  return allParameters
def subsetParameters(metricsArray):
  
  print(metricsArray[:,0])
  return 'done'

###########################
### CALIBRATE THE MODEL ###
###########################
 
print("Parameter values are stored in 3 dimensional array [timestep, iteration, metric]")
print("# timestep: year \n# iteration: set of parameters used \n# metric: array of values \
of the selected metric for each zone for each set of parameters seperately")


# Calibration of the modell will be based on finding
# minimum root-mean-square error between the metrics modelled and observed.

for aVariable in metricList:
  zonesModelled = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))
  
  # 1. Create a list containing possible paramter configurations
  allParameters = getParameterConfigurations(zonesModelled)
  noParameterConfigurations = zonesModelled.shape[1]
  
  # 2. Create an array to store the modelled metrics values for the observation time steps
  modelledArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,16,1))# change this!!!!! to number of zones
  differenceArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,16,1))# change this!!!!!
  obsArray = np.zeros((len(obsTimeSteps),1,16,1))# change this!!!!!
  
  # Store the metric value for selected timestep
  for obsTimeStep in obsTimeSteps:
    rowIndex = 0
    for p in range(0,noParameterConfigurations):
      modelledArray[rowIndex,p] = zonesModelled[obsTimeStep-1,p][1]
    rowIndex = rowIndex+1
  
  # 3. Create an array to store the observed metrics values
  zonesObserved = np.load(os.path.join("results", country, 'metrics', aVariable + '_obs.npy'))
  rowIndex = 0
  for obsTimeStep in range(0,len(obsTimeSteps)):
    obsArray[rowIndex,0] = zonesObserved[obsTimeStep,0][1]
    rowIndex = rowIndex+1
  
  # 4. Create difference array
  differenceArray = numpy.subtract(modelledArray,obsArray)
  print(differenceArray)
  
  


  


# 1. Calculate Mean Squared Error
numberOfDataPoints = 1 # number of observations. Here: 2000

# Get the modelled values corresponding to the observed values

'''print(zonesModelled)
  metricValueArray = [] # [metric name, parameter set, timestep, zone, metric value]
  metricsArray = [] # array to store matricValueArray[] for each zone for each timestep for each metric
  
  for timeStep in timeSteps: # Loop data for each time step
    for zone in range(0, len(zonesModelled[timeStep-1][0][1])): # Loop array to extraxt data for each zone
      for i in range(0,len(zonesModelled[timeStep-1])): # get the metric for the time step in given zone
        for obsTimeStep in obsTimeSteps:
          if timeStep == obsTimeStep:
            metricValueArray = [aVariable,zonesModelled[timeStep-1][i][0],timeStep,zone,zonesModelled[timeStep-1][i][1][zone][0]]
            metricsArray.append(metricValueArray)
            metricValueArray = []'''






  
  
  


