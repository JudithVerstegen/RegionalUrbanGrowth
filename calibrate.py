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
#variables = parameters.getSumStats() 
variables = ['np'] # FOR TESTING

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps=parameters.getNrTimesteps()
numberOfIterations = parameters.getNumberofIterations(parameters.getSuitFactorDict(), parameters.getParametersforCalibration())

iterations = range(1, numberOfIterations+1, 1)
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsSampleNumbers = [1] #range(1,20+1,1) <- for stochastic model
obsTimeSteps = [10,16] # for a whole run should be amended # year 2000 and 2006. maybe should go to parameters?

# Path to the folder with the metrics npy arrays stored
country = parameters.getCountryName()
arrayFolder = os.path.join(os.getcwd(),'results',country,'metrics')


#################
### FUNCTIONS ###
#################



###########################
### CALIBRATE THE MODEL ###
###########################
 
print("Parameter values are stored in 3 dimensional array [timestep, iteration, metric]")
print("# timestep: year \n# iteration: set of parameters used \n# metric: array of values \
of the selected metric for each zone for each set of parameters seperately")


# Calibration of the modell will be based on finding
# minimum root-mean-square error between the metrics modelled and observed.

# 1. Calculate Mean Squared Error
numberOfDataPoints = 2 # number of observations. Here: 2000 and 2006

# Get the modelled values corresponding to the observed values
for aVariable in variables:
  zonesModelled = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))
  print(zonesModelled[0])
  metricValueArray = [] # [metric name, parameter set, timestep, zone, metric value]
  metricsArray = [] # array to store matricValueArray[] for each zone for each timestep for each metric
  

  for timeStep in timeSteps: # Loop data for each time step
    
    for zone in range(0, len(zonesModelled[timeStep-1][0][1])): # Loop array to extraxt data for each zone

      for i in range(0,len(zonesModelled[timeStep-1])): # get the metric for the time step in given zone
        for obsTimeStep in obsTimeSteps:
          if timeStep == obsTimeStep:
            metricValueArray = [aVariable,zonesModelled[timeStep-1][i][0],timeStep,zone,zonesModelled[timeStep-1][i][1][zone][0]]
            metricsArray.append(metricValueArray)
            metricValueArray = []

# Create a list of parametr configurations
allParameters = []
for i in range(0,len(metricsArray)):
  allParameters.append(metricsArray[i][1])

# Convert array to list
allParameters = np.array(allParameters).tolist()
# Find duplicates
paramList = []
for elem in allParameters:
    if elem not in paramList:
        paramList.append(elem)
print(paramList)

for paramSet in paramList:
  print(paramSet)
  if (metricsArray[1][1] == paramSet).all():
    print('.')






  
  
  


