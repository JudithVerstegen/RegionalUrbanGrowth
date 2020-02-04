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
obsTimeSteps = parameters.getObsTimesteps() 

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
  
  # Create a 3D array: no of rows = no of observed timesteps, no of columns = no of parameter sets
  # Each cell store: the parameter set and the values of zones
  theArray = np.zeros((len(obsTimeSteps),noParameterConfigurations,numberZones,1))
  
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,noParameterConfigurations):
      for zone in range(0,numberZones):
        theArray[row,col][zone][0] = modelled[obsTimeSteps[row]-1,col][1][zone][0] - observed[row,0][1][zone][0]  
  return theArray

def calculateKappa():
  # Create arrays to store the metrices by Pontius (quantity disagreement and allocation disagreement)
  quantityArray = np.zeros((len(obsTimeSteps),numberOfIterations))
  allocationArray = np.zeros((len(obsTimeSteps),numberOfIterations))
  # Create array to store transition states:
  transArray = np.zeros((len(obsTimeSteps),numberOfIterations,3,3))
  # Create array to store Kappa
  kappaArray = np.zeros((len(obsTimeSteps),numberOfIterations))
  # Create array to store Kappa Simulation (by Van Vliet):
  kappaSimulation = np.zeros((len(obsTimeSteps),numberOfIterations))
    
  # Load data:
  urbObs = getObservedArray('urb')
  cells = len(urbObs[0,0,1])
  nanCells = sum(np.isnan(urbObs[0,0,1]))
  validCells = cells - nanCells
      
  # Loop observed years:
  for row in enumerate(obsTimeSteps):
    print(row)
    # Load modelled urban for the given obs year
    urbMod = np.load(os.path.join(arrayFolder, 'urb_subset_'+str(row[1]) + '.npy'))

    # Define conditions to compare simulated and actual (observed) maps
    obs1 = (urbObs[row[0],0,1] == 1)
    obs0 = (urbObs[row[0],0,1] == 0)
    
    # Loop parameter sets:
    for col in range(0,numberOfIterations):
      # Create contingency table, full of zeros (to calculate Kappa)
      cArray = np.zeros((3,3))
      # Create population table, full of zeros (to calculate quantity disagreement and allocation disagreement)
      popMatrix = np.zeros((3,3))
        
      # For year 1990 (first observation time step) there is a perfect agreement:
      if row[0] == 0:
        kappaArray[row[0],col] = 1.0
        kappaSimulation[row[0],col] = 1.0
        transArray[row[0],col] = cArray
        
      else:
        # Define conditions to compare simulated and actual (observed) maps
        mod1 = (urbMod[0,col,1] == 1)
        mod0 = (urbMod[0,col,1] == 0)

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

        # Fill the contingency table (Pontius and Millones, 2011, Table 2):
        cArray[0,0] = sum(allStates == 1)/validCells
        cArray[1,0] = sum(allStates == 2)/validCells
        cArray[0,1] = sum(allStates == 3)/validCells
        cArray[1,1] = sum(allStates == 4)/validCells
        cArray[2,0] = (sum(allStates == 1)+sum(allStates == 2))/validCells
        cArray[2,1] = (sum(allStates == 3)+sum(allStates == 4))/validCells
        cArray[0,2] = (sum(allStates == 1)+sum(allStates == 3))/validCells
        cArray[1,2] = (sum(allStates == 2)+sum(allStates == 4))/validCells
        cArray[2,2] = 1
        
        transArray[row[0],col] = cArray

        # Calculate quantity disagreemnt (Pontius and Millones, 2011) (2)
        q_urban = np.absolute(cArray[0,0]+cArray[1,0]-(cArray[0,0]+cArray[0,1]))
        q_non_urban = np.absolute(cArray[0,1]+cArray[1,1]-(cArray[1,0]+cArray[1,1]))
        Q = (q_urban + q_non_urban) / 2 # (3)

        # Calculate allocation disagreement (Pontius and Millones, 2011): (4)
        a_urban = 2 * np.minimum(cArray[1,0], cArray[0,1])
        a_non_urban = 2 * np.minimum(cArray[0,1], cArray[1,0])
        A = (a_urban + a_non_urban) / 2 # (5)

        # Calculate the proportion correct
        C = cArray[0,0] + cArray[1,1] # (6)
        
        # Calculate the total disagreement
        D = 1 - C # = Q + A (7)

        # Calculate fractions of agreement:
        E = cArray[0,2]*cArray[2,0] + cArray[1,2]*cArray[2,1] # (8,9)
        
        # Calculate Kappa
        K_standard = (C - E) / (1 - E) # (11)
        print('year: ', str(row[1]),', parameter set: ', str(col), ', Kappa standard:',K_standard)
        
        # Save Kapa in array
        kappaArray[row[0],col] = K_standard
        
        # Save errors
        quantityArray[row[0],col] = Q
        allocationArray[row[0],col] = A

  # Save all the metrices
  metrices = {
    'kappa': kappaArray,
    'quantity_disagreement': quantityArray,
    'allocation_disagreement': allocationArray,
    'transition_matrix': transArray
    }
  for name in metrices.keys():
    # Set the name of the file
    fileName = os.path.join(arrayFolder, str(name))

    # Clear the directory if needed
    if os.path.exists(fileName + '.npy'):
        os.remove(fileName + '.npy')

    # Save the data  
    np.save(fileName, metrices[name])
    
  print('Kappa and Pontius metrices calculated and saved as npy files')

def calculateKappaSimulation():
  # Create array to store Kappa Simulation (by Van Vliet):
  kappaSimulation = np.zeros((len(obsTimeSteps),numberOfIterations))
    
  # Load data:
  urbObs = getObservedArray('urb')
  
  cells = len(urbObs[0,0,1])
  nanCells = sum(np.isnan(urbObs[0,0,1]))
  validCells = cells - nanCells

  # Define conditions (urban and non-urban cells) in the original map (1990):
  original_0 = (urbObs[0,0,1] == 0) # TRUE

  original_1 = (urbObs[0,0,1] == 1) # TRUE
  p_org0 = sum(original_0)/validCells
  p_org1 = sum(original_1)/validCells
      
  # Loop observed years:
  for row in enumerate(obsTimeSteps):
    print(row)
    # Load modelled urban for the given obs year
    urbMod = np.load(os.path.join(arrayFolder, 'urb_subset_'+str(row[1]) + '.npy'))

    # Define conditions to compare simulated and actual (observed) maps
    obs1 = (urbObs[row[0],0,1] == 1)
    obs0 = (urbObs[row[0],0,1] == 0)
    # Define conditions and calculate the transitions:
    act_0_1 = np.where(obs1 & original_0,1,0)
    act_1_0 = np.where(obs0 & original_1,1,0)
    p_act_0_1 = sum(act_0_1)/validCells
    p_act_1_0 = sum(act_1_0)/validCells
    
    # Loop parameter sets:
    for col in range(0,len(urbMod[0])):
       
      # For year 1990 (first observation time step) there is a perfect agreement:
      if row[0] in [0]:
        kappaSimulation[row[0],col] = 1.0
        
      else:
        # Define conditions to compare simulated and actual (observed) maps
        mod1 = (urbMod[0,col,1] == 1)
        mod0 = (urbMod[0,col,1] == 0)

        # Calculate P0 between simulated and actual map (by Van Vliet):
        PO_urban = sum(obs1 & mod1)/validCells
        PO_non_urban = sum(obs0 & mod0)/validCells
        PO = PO_urban + PO_non_urban

        # Define conditions to compare transitions in original and simulated maps
        sim_0_1 = np.where(mod1 & original_0,1,0)
        sim_1_0 = np.where(mod0 & original_1,1,0)
        
        # Calculate PE Transition (Van Vliet, 2013, Eq. 2.7):
        p_sim_0_1 = sum(sim_0_1)/validCells
        p_sim_1_0 = sum(sim_1_0)/validCells
        
        PE_transition = (p_org0 + p_org1) * (p_act_0_1 * p_sim_0_1 + p_act_1_0 * p_sim_1_0)

        # Calculate Kappa Simulation (Van Vliet, 2013, Eq. 2.9):
        K_simulation = (PO - PE_transition) / (1 - PE_transition)
        
        print('year: ', str(row[1]),', parameter set: ', str(col), ', Kappa simulation:',K_simulation)
        
        # Save Kappa Simulation in array
        kappaSimulation[row[0],col] = K_simulation
      
  # Set the name of the file
  fileName = os.path.join(arrayFolder, str('kappa_simulation'))

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, kappaSimulation)

calculateKappaSimulation()

def calcRMSE(diffArray, aVariable, scenario=None):
  pSets = diffArray.shape[1]
  zones = diffArray.shape[2]
  # Create empty array. Rows = nr of observed timesteps, columns = nr of parameter sets
  rmseArray = np.zeros((diffArray.shape[0],diffArray.shape[1]))

  # Calculate RMSE for each timestep and parameter set. Number of observations = number of zones
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,pSets):
      # Create a list containing values for each zone or selected zones only
      x = diffArray[row,col].flatten()
      if scenario==3:
        # Path to the file storing numbers of calibration zones:
        calibration_zones_file = os.path.join(os.getcwd(),'input_data', country, 'calibration_zones.txt')
        # Read only first column from the file and get the zones numbers.
        # The zone numbers start with 1, so substract 1 to get values starting from 0:
        calibration_zones = [int(x.split("\t")[0])-1 for x in open(calibration_zones_file).readlines()]
        x = np.array([diffArray[row,col].flatten()[x] for x in calibration_zones])
      # Remove nan values
      x = x[~numpy.isnan(x)]
      # Calculate RMSE for each zones
      rmseArray[row,col] = np.sqrt(np.mean(x**2))

  return rmseArray
  
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

def RMSEindex_selectedPeriod(aRMSE,scenario,aim):
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

def RMSEindex_selectedArea(metric,aim):
  if metric in ['cilp','pd']:
    metric = metric + '_' + aim[:3]
    
  modelled = getModelledArray(metric)
  observed = getObservedArray(metric)
  dArray = createDiffArray(modelled,observed)
  # For the metrics calculated for each zone separately (fdi, wfdi) calculate the RMSE only for selected zones
  if metric in ['fdi', 'wfdi']: 
    zoneRMSE = calcRMSE(dArray, metric, scenario=3)
  # For 'pd' and 'cilp' the selected zones are already saved in the files
  else:
    zoneRMSE = calcRMSE(dArray, metric)

  return RMSEindex_selectedPeriod(zoneRMSE,3,'calibration')
  
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

 
def RMSEindex_KAPPAindex_selectedPeriod(aRMSE,aKAPPA,w_RMSE,w_Kappa,scenario,aim):
  # multi - objective goal function
  # RMSE and kappa metrics are combined (summed)
  # each metric has a weight (w1,w2)
  w1 = w_RMSE
  w2 = w_Kappa
  # scenario and aim define the period, for which the RMSE is to be calculated
  # period is a list containing indexes of selected years, it is defined by the scenario
  period = parameters.getCalibrationPeriod()[scenario][aim]
  # Normalize the mean RMSE values for the selected period
  RMSE_mean = [(r1+r2)/2 for r1,r2 in zip(aRMSE[period][0],aRMSE[period][1])]
  RMSE_max = np.amax(RMSE_mean)
  RMSE_min = np.amin(RMSE_mean)
  RMSE_norm = [(RMSE_max - x) / (RMSE_max - RMSE_min) for x in RMSE_mean]

  # Calculate the mean value for Kappa.It has already values 0-1, co there is no need for normalization.
  Kappa_mean = [(k1+k2)/2 for k1,k2 in zip(aKAPPA[period][0],aKAPPA[period][1])]
  Kappa_max = np.amax(Kappa_mean)
  Kappa_min = np.amin(Kappa_mean)
  Kappa_norm = [(x - Kappa_min) / (Kappa_max - Kappa_min) for x in Kappa_mean]
  
  rBase = [RMSE_norm[0],0]
  kBase = [Kappa_norm[0],0]
  
  multiBase = [w1 * rBase[0] + w2 * kBase[0],0] # the bigger the better
  # Loop parameter configurations to find the one with the best goal function outcome for the selected period
  for p in range(len(aKAPPA[0,:])): 
    # select normalized mean RMSE for the given parameter
    mRMSE = RMSE_norm[p]
    # calculate mean kappa for the given parameter
    mKappa = Kappa_norm[p]
    
    multiGoal = w1 * mRMSE + w2 * mKappa
    if multiGoal > multiBase[0]:
      multiBase[0] = multiGoal
      multiBase[1] = p
    
  return multiBase[1]

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

  #x = np.concatenate(('Calibration:',c['calibration'],'Validation:'],c['validation']), axis=0)
  x = ['Calibration:']+list(c['calibration'])+['Validation:']+list(c['validation'])
  return x


############33 TESTTT
'''kappa = np.load(os.path.join(arrayFolder,'kappa.npy'))
print(kappa)

timestep=29
theName = 'urb' + str(timestep) + '.obj'
fileName = os.path.join('F:/results/IE', str(158), theName)
filehandler = open(fileName, 'rb')

data = pickle.load(filehandler)
urbObs = getObservedArray('urb')
#print('observed',sum(urbObs[4,0,1]==1))
#print('modelled from pickle:', sum(data==1))
setclone('F:/input_data/IE/nullmask.map')
for row in enumerate(obsTimeSteps):
  print(row)
  if(row[1]==29):
    urbMod = np.load(os.path.join(arrayFolder, 'urb_subset_'+str(row[1]) + '.npy'))
    np.set_printoptions(threshold=np.inf)
    urbMod159year2018 = urbMod[0,158,1].reshape((1600,1600))
    n2p = numpy2pcr(Nominal,urbMod159year2018,0)
    aguila(n2p)


'''
'''
ALMOST DONE
multiobjective = np.zeros((12,11))
kappa = np.load(os.path.join(arrayFolder,'kappa.npy'))
k=[]
for k1, k2 in zip(kappa[3:5][0],kappa[3:5][1]):
  k.append((k1+k2)/2)
for w in range(0,11,1):
  rmse_weight = w/10
  kappa_weight = 1 - rmse_weight
  #print('rmse ',rmse_weight,'kappa ',kappa_weight)
        
  for row,metric in enumerate(['cilp','fdi','wfdi','pd']):
    modelled = getModelledArray(metric)
    observed = getObservedArray(metric)  
    dArray = createDiffArray(modelled,observed)
    rmse = calcRMSE(dArray, metric)
    theR = (rmse[3:5][0]+rmse[3:5][1])/2
    

    hereIndex = RMSEindex_KAPPAindex_selectedPeriod(rmse,kappa,rmse_weight,kappa_weight,1,'calibration')
    multiobjective[row*3, w] = hereIndex
    multiobjective[row*3+1, w] = theR[hereIndex]
    multiobjective[row*3+2, w] = k[hereIndex]
print('done')
saveResults(multiobjective, 1, 'multiobjective.csv')
'''
