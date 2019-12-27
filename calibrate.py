''' Calibration phase of the LU_urb.py '''

import pickle
import os
import metrics
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt
import csv

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
  
def getParameterConfigurations():
  return np.load(os.path.join(arrayFolder, 'parameter_sets.npy'))

def createDiffArray(modelled,observed):
  # Get the number of parameter configurations
  noParameterConfigurations = modelled.shape[1]
  #print(modelled)
  # Get the number of zones
  numberZones = len(modelled[0][0][1])
  print(numberZones)
  # Create a 3D array: no of rows = no of observed timesteps, no of columns = no of parameter sets
  # Each cell store: the parameter set and the values of zones
  theArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,numberZones,1))
  print(theArray)
  
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,noParameterConfigurations):
      for zone in range(0,numberZones):
        theArray[row,col][zone][0] = modelled[obsTimeSteps[row]-1,col][1][zone][0] - observed[row,0][1][zone][0]  
  return theArray

  
def calculateKappa():
  # Create an array to store the comparison between the observed and modelled urban areas.
  
  # Create array to store transition states:
  transArray = np.zeros((len(obsTimeSteps),numberOfIterations,3,3))
  # Create array to store Kappa
  kappaArray = np.zeros((len(obsTimeSteps),numberOfIterations))

  
  # Load data:
  urbObs = getObservedArray('urb')
  urbMod = getModelledArray('urb_subset')
  cells = len(urbObs[0,0,1])
  nanCells = sum(np.isnan(urbObs[0,0,1]))
  validCells = cells - nanCells
  print('valid:', validCells)
      
  # Loop years:
  for row in range(len(obsTimeSteps)):
    
    # Loop parameter sets:
    for col in range(0,numberOfIterations):
      # Create contingency table, full of zeros
      cArray = np.zeros((3,3))
        
      # For year 1990 (row 0) there is a perfect agreement:
      if row == 0:
        kappaArray[row,col] = 1.0
        transArray[row,col] = cArray
        
      else:
        # Define conditions
        obs1 = (urbObs[row,0,1] == 1)
        obs0 = (urbObs[row,0,1] == 0)
        mod1 = (urbMod[row,col,1] == 1)
        mod0 = (urbMod[row,col,1] == 0)

        # Find states. Each cell has only one of four states.
        # Observed -> modelled
        # 1. urban -> urban
        # 2. urban -> non-urban
        # 3. non-urban -> urban
        # 4. non-urban -> non-urban
        state1 = np.where(obs1 & mod1,1,0)
        state2 = np.where(obs1 & mod0,2,0)
        state3 = np.where(obs0 & mod1,3,0)
        state4 = np.where(obs0 & mod0,4,0) 

        # Save state of each cell into the array
        allStates = state1+state2+state3+state4

        # Fill the contingency table:
        cArray[0,0] = sum(allStates == 1)/validCells
        cArray[0,1] = sum(allStates == 2)/validCells
        cArray[1,0] = sum(allStates == 3)/validCells
        cArray[1,1] = sum(allStates == 4)/validCells
        cArray[0,2] = (sum(allStates == 1)+sum(allStates == 2))/validCells
        cArray[1,2] = (sum(allStates == 3)+sum(allStates == 4))/validCells
        cArray[2,0] = (sum(allStates == 1)+sum(allStates == 3))/validCells
        cArray[2,1] = (sum(allStates == 2)+sum(allStates == 4))/validCells
        cArray[2,2] = 1

        transArray[row,col] = cArray

        # Calculate fractions of agreement:
        P0 = cArray[0,0] + cArray[1,1]
        PE = cArray[0,2]*cArray[2,0] + cArray[1,2]*cArray[2,1]
        PMAX = np.minimum(cArray[0,2],cArray[2,0]) + np.minimum(cArray[1,2],cArray[2,1])

        # Calculate Kappa
        Kappa = (P0 - PE) / (1 - PE)
        
        # Save Kapa in array
        kappaArray[row,col] = Kappa
        #print(cArray)

  # Set the name of the file
  fileName = os.path.join(arrayFolder, 'kappa')

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, kappaArray)
  print('Kappa calculated and saved as kappa.npy')

def calcRMSE(diffArray, aVariable, scenario=None):
  pSets = diffArray.shape[1]
  zones = diffArray.shape[2]
  # Create empty array. Rows = nr of observed timesteps, columns = nr of parameter sets
  rmseArray = np.zeros((diffArray.shape[0],diffArray.shape[1]))

  # Calculate RMSE for each timestep and parameter set. Number of observations = number of zones
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,pSets):
      # Create a list containing values for each zone
      x = diffArray[row,col].flatten()
      # Remove nan values
      x = x[~numpy.isnan(x)]
      # Calculate RMSE for each zones
      rmseArray[row,col] = np.sqrt(np.mean(x**2))

  return rmseArray



#########
"""
for m in ['pd']:
    zonesModelled = getModelledArray(m)
    zonesObserved = getObservedArray(m)
    diffArray = createDiffArray(zonesModelled,zonesObserved)
    rmseArray = calcRMSE(diffArray, m, 3)
    RMSEindex_selectedArea(rmseArray,3,'calibration')
    """
##########


    
def RMSEindex_eachTimeStep(rmseArr):
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

def RMSEindex_selectedPeriod(aRMSE,scenario,aim): #smallestMeanErrorIndex
  # scenario and aim define the period, for which the RMSE is to be calculated
  # period is a list containing indexes of selected years, it is defined by the calibration scenario

  period = parameters.getCalibrationPeriod()[scenario][aim]
  mBase = [np.absolute(np.mean(aRMSE[period,0])),0]
  # Loop parameter configurations to find the one with the smalles abs mean value for all years
  for p in range(len(aRMSE[0,:])):
    mRMSE = np.absolute(np.mean(aRMSE[period,p]))
    if mRMSE < mBase[0]:
      mBase[0] = mRMSE
      mBase[1] = p
  return mBase[1]

#def smallestMeanErrorIndex_2000_2006(theRMSE): # to be removed and included in RMSEindex_selectedPeriod()

#def smallestMeanErrorIndex_2012_2018(theRMSE): # to be removed and included in RMSEindex_selectedPeriod()

def RMSEindex_selectedArea(theRMSE, scenario, aim):
  mask_zones = parameters.getCalibrationArea()[aim]
  period = parameters.getCalibrationPeriod()[scenario][aim]
  #mask =
  mBase = [np.absolute(np.mean(theRMSE[period,0])),0]
  
  # Loop parameter configurations to find the one with the smalles abs mean value for zones 0-7
  for p in range(len(theRMSE[0,:])):
    mRMSE = np.absolute(np.mean(theRMSE[:,p]))
    if mRMSE < mBase[0]:
      mBase[0] = mRMSE
      mBase[1] = p
  return mBase[1]

def biggestKappaInObsYear(kappaArr):
  fitList = []
  
  for year in range(1,len(obsTimeSteps)):
    index = 0
    base = [kappaArr[year,:][0],index]
    for r in kappaArr[year,:]:
      if r > base[0]:
        base[0] = r
        base[1] = index
      index = index+1
    # Create a list storing a liast of observed timestep and index of parameter set)
    fitList.append([obsTimeSteps[year],base[1]])
  return(fitList)

def biggestKappa_2000_2006(kappaArr):
  kBase = [np.mean(kappaArr[1:3,0]),0]
  # Loop parameter configurations to find the one with the best mean value for years 2000 and 2006
  for p in range(len(kappaArr[0,:])):
    mKappa = np.mean(kappaArr[1:3,p])
    if mKappa > kBase[0]:
      kBase[0] = mKappa
      kBase[1] = p
  return kBase[1]

def biggestKappa_2012_2018(kappaArr):
  kBase = [np.mean(kappaArr[-2,0]),0]
  # Loop parameter configurations to find the one with the best mean value for years 2000 and 2006
  for p in range(len(kappaArr[0,:])):
    mKappa = np.mean(kappaArr[-2,p])
    if mKappa > kBase[0]:
      kBase[0] = mKappa
      kBase[1] = p
  return kBase[1]

'''####### Testing for combinaing kappa and single metric:
def normalized(a, axis=-1, order=2): ## <-- what is this thing doing
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
  
def testKasi(KAPPAarr, RMSEarr):
  RMSEarr = normalized(RMSEarr)
  kBase = [np.mean(KAPPAarr[1:3,0]),0]
  mBase = [np.absolute(np.mean(RMSEarr[1:3,0])),0]
  testBase = [np.mean([1-kBase[0], mBase[0]]),0] # the smaller the better
  # Loop parameter configurations to find the one with the best mean value for years 2000 and 2006
  for p in range(len(KAPPAarr[0,:])):
    mKappa = np.mean(KAPPAarr[1:3,p])
    mRMSE = np.absolute(np.mean(RMSEarr[1:3,p]))
    mTest = np.mean([1-mKappa, mRMSE])
    if mTest < testBase[0]:
      testBase[0] = mTest
      testBase[1] = p
    
  return testBase[1]

######## <------- testing!'''

###############################################################################################################
############################ CALBRATE AND VALIDATE
###############################################################################################################

def getCalibratedParameters(calibrationScenario):
  # create a list of indexes
  indexes = []
  for m in metricList:
    zonesModelled = getModelledArray(m)
    zonesObserved = getObservedArray(m)
    rmseArray = calcRMSE(createDiffArray(zonesModelled,zonesObserved), m)
    if calibrationScenario in [1,2]:
      calibratedIndex = RMSEindex_selectedPeriod(rmseArray, calibrationScenario,'calibration')
    elif calibrationScenario == 3:
      calibratedIndex = RMSEindex_selectedArea(rmseArray, 'calibration')
    else:
      print('Invalid calibration scenario. Possible scenarios: 1,2,3')
    
    parameterSets = list(getParameterConfigurations())
    theParameters = list(parameterSets[calibratedIndex])
    theParameters.insert(0,calibratedIndex)
    theParameters.insert(0,m)
    indexes.append(theParameters)
  
  kappaArray = np.load(os.path.join(arrayFolder, 'kappa.npy'))
  kappaIndexes = {
    1: biggestKappa_2000_2006(kappaArray),
    2: biggestKappa_2012_2018(kappaArray)
    }
  kappaIndex = kappaIndexes[calibrationScenario] 
  kappaParameters = list(parameterSets[kappaIndex])
  kappaParameters.insert(0,kappaIndex)
  kappaParameters.insert(0,'kappa')
  indexes.append(kappaParameters)

  return indexes

def saveResults(array, scenario, fileName):
  with open(os.path.join(arrayFolder,'Scenario'+'_'+str(scenario)+'_'+fileName),
            'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for row in array:
      writer.writerow(row)

def calibrate_validate(scenario): 
  indexes = np.array(getCalibratedParameters(scenario))[:,1]
  kappaArray = np.load(os.path.join(arrayFolder, 'kappa.npy'))

  c={}
  for aim in ['calibration', 'validation']:
    aim_period = parameters.getCalibrationPeriod()[scenario][aim]
    # create the array to store the calibration values
    v = np.zeros((len(metricList)+1,len(metricList)+1))
    for i, index in enumerate(indexes):
      for j, metricCol in enumerate(metricList+['kappa']):
        if metricCol == 'kappa':
          # Calculate the mean Kappa for the selected scenario
          v[i,j] = np.mean(kappaArray[aim_period, int(index)])
        else:
          # Calculate the mean RMSE for the selected senario
          zonesModelled = getModelledArray(metricCol)
          zonesObserved = getObservedArray(metricCol)
          rmseArray = calcRMSE(createDiffArray(zonesModelled,zonesObserved), metricCol)
          v[i,j] = np.mean(rmseArray[aim_period, int(index)])
    c[aim] = v

  x = np.concatenate((c['calibration'],c['validation']), axis=0)
  return x
   
'''def validate(calibrationScenario): 
  indexes = np.array(getCalibratedParameters(calibrationScenario))[:,1]
  kappaArray = np.load(os.path.join(arrayFolder, 'kappa.npy'))
  
  scenarioPeriod = {
    1: -2,
    2: [1,2]
    }
  
  # create the array to store the calibration values
  v = np.zeros((len(metricList)+1,len(metricList)+1))
  for i, index in enumerate(indexes):
    for j, metricCol in enumerate(metricList+['kappa']):
      if metricCol == 'kappa':
        # Calculate the mean Kappa for the selected senario
        v[i,j] = np.mean(kappaArray[scenarioPeriod[calibrationScenario], int(index)])
      else:
        # Calculate the mean RMSE for the selected senario
        zonesModelled = getModelledArray(metricCol)
        zonesObserved = getObservedArray(metricCol)
        rmseArray = calcRMSE(createDiffArray(zonesModelled,zonesObserved), metricCol)
        v[i,j] = np.mean(rmseArray[scenarioPeriod[calibrationScenario], int(index)])
  return v'''


