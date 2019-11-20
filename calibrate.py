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

def getModelledArray(metric):
  return np.load(os.path.join(arrayFolder, metric + '.npy'))

def getObservedArray(metric):
  return np.load(os.path.join(arrayFolder, metric + '_obs.npy'))
  

def getParameterConfigurations(metricsArray):
  allParameters = []
  for i in range(0,len(metricsArray[0])):
    allParameters.append(metricsArray[0][i][0].tolist())
  return allParameters

def createDiffArray(modelled,observed):
  # Get the number of parameter configurations
  noParameterConfigurations = modelled.shape[1]
  # Get the number of zones
  numberZones = len(modelled[0][0][1])
  # Create a 3D array: no of rows = no of observed timesteps, no of columns = no of parameter sets
  # Each cell store: the parameter set and the values of zones
  theArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,numberZones,1))
  
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,noParameterConfigurations):
      for zone in range(0,numberZones):
        #theArray[row,col][zone][0] = [999]#modelled[obsTimeSteps[0]-1,0][0]
        theArray[row,col][zone][0] = modelled[obsTimeSteps[row]-1,col][1][zone][0] - observed[row,0][1][zone][0]  
  return theArray
  
def calcRMSE(diffArray, aVariable):
  pSets = diffArray.shape[1]
  zones = diffArray.shape[2]
  # Create empty array. Rows = nr of observed timesteps, columns = nr of parameter sets
  rmseArray = np.zeros((diffArray.shape[0],diffArray.shape[1]))

  # Calculate RMSE for each timestep and parameter set. Number of observations = number of zones
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,pSets):
      rmseArray[row,col] = np.sqrt(np.mean((diffArray[row,col].flatten())**2))

  return rmseArray

def smallestErrorInObsYear(rmseArr):
  fitList = []
  
  for year in range(1,len(obsTimeSteps)):
    index = 0
    base = [rmseArr[year,:][0],index]
    for r in rmseArr[year,:]:
      if np.absolute(r) < np.absolute(base[0]):
        base[0] = r
        base[1] = index
      index = index+1
    # Create a list storing a liast of observed timestep and index of parameter set)
    fitList.append([obsTimeSteps[year],base[1]])
  return(fitList)

def smallestMeanErrorIndex(aRMSE):
  mBase = [np.absolute(np.mean(aRMSE[:,0])),0]
  # Loop parameter configurations to find the one with the smalles abs mean value for all years
  for p in range(len(aRMSE[0,:])):
    mRMSE = np.absolute(np.mean(aRMSE[:,p]))
    if mRMSE < mBase[0]:
      mBase[0] = mRMSE
      mBase[1] = p
  return mBase[1]

def smallestMeanErrorIndex_2000_2006(theRMSE):
  mBase = [np.absolute(np.mean(theRMSE[1:3,0])),0]
  # Loop parameter configurations to find the one with the smalles abs mean value for years 2000 and 2006
  for p in range(len(theRMSE[0,:])):
    mRMSE = np.absolute(np.mean(theRMSE[1:3,p]))
    if mRMSE < mBase[0]:
      mBase[0] = mRMSE
      mBase[1] = p
  return mBase[1]

def saveTheArray(output, metricName, fileEnd): 
  # Set the name of the file
  fileName = os.path.join(arrayFolder, metricName + fileEnd)

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
    os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, output)

def calculateKappa():
  print('done.')
  
###########################
### CALIBRATE THE MODEL ###
###########################
 
print("CALIBRATE THE MODEL")
print('Observation time steps:',obsTimeSteps)
print('.')


# Calibration of the modell will be based on finding
# minimum root-mean-square error between the metrics modelled and observed.

for aVariable in metricList:
  print('Metric: ',aVariable)
  zonesModelled = getModelledArray(aVariable)
  zonesObserved = getObservedArray(aVariable)

  # 0. Get the parameter sets
  parameterSets = getParameterConfigurations(zonesModelled)

  # 1. Create an array storing differences between the modelled and observed values
  dArray = createDiffArray(zonesModelled,zonesObserved)

  # 2. Calculate Root Mean Squared Error (RMSE)
  RMSE = calcRMSE(dArray, aVariable)
  saveTheArray(RMSE, aVariable, '_RMSE')

  # 3. Find the parameter sets with the smallest RMSE
  print('Smallest RMSE in each year parameter sets [year time step, set]:',smallestErrorInObsYear(RMSE))
  print('Smallest mean RMSE parameter set:',smallestMeanErrorIndex(RMSE))

  # 4. Find the fitting parameter set for calibration for years 2000 and 2006
  fittingSet = smallestMeanErrorIndex_2000_2006(RMSE)
  print('Smallest mean RMSE for 2000 (timestep 11) and 2006 (timestep 17) parameter set:', fittingSet,
        parameterSets[fittingSet])

  # 5. Calculate Kappa statistic


