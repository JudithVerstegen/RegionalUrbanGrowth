''' Calibration phase of the LU_urb.py '''

import pickle
import os
import metrics
import numpy as np
import parameters
from pcraster.framework import *
from numba import njit
from scipy.spatial import distance

#### Script to find calibrate LU model

# Work directory:
work_dir = parameters.getWorkDir()

# Get metrics
metricList = parameters.getSumStats()
locationalMetric = parameters.getLocationalAccuracyMetric()
all_metrices = metricList+locationalMetric
# Get case studies
case_studies = parameters.getCaseStudies()
# Get calibration scenarios
scenarios = parameters.getCalibrationScenarios()

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps=parameters.getNrTimesteps()
numberOfIterations = parameters.getNumberofIterations()
iterations = range(1, numberOfIterations+1, 1)
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsSampleNumbers = [1] #range(1,20+1,1) <- for stochastic model
obsTimeSteps = parameters.getObsTimesteps() 

# Path to the folder with the metrics npy arrays stored
country = parameters.getCountryName()
arrayFolder = os.path.join(work_dir,'results',country,'metrics')

#################
### FUNCTIONS ###
#################

def getModelledArray(metric,scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics')
  return np.load(os.path.join(folder, metric + '.npy'),allow_pickle=True)

def getObservedArray(metric,scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics')
  return np.load(os.path.join(folder, metric + '_obs.npy'),allow_pickle=True)

def getSubsetArray(metric,index,scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics')
  return np.load(os.path.join(folder, metric+'_subset_' + str(index) + '.npy'),allow_pickle=True)
  
def getParameterConfigurations(case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics')
  return np.load(os.path.join(folder, 'parameter_sets.npy'))

def getKappaArray(scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics') 
  return np.load(os.path.join(folder, 'kappa.npy'))

def getKappaSimulationArray(aim=None,scenario=None, case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics')
  return np.load(os.path.join(folder, 'kappa_simulation.npy'))

def getAllocationArray(aim=None,scenario=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(work_dir,'results',case,'metrics')  
  return np.load(os.path.join(folder, 'allocation_disagreement.npy'))

def getAveragedArray(array, scenario, aim):
  # period is a list containing indexes of selected years, it is defined by the scenario
  period = parameters.getCalibrationPeriod()[scenario][aim]
  # get average values
  a_mean = np.mean(array[period],axis=0)
  return a_mean

def getNormalizedArray(array, kappa=None):
  a_max = np.amax(array)
  a_min = np.amin(array)
  if kappa==True: 
    a_norm = [(x - a_min) / (a_max - a_min) for x in array]
  else:
    a_norm = [(a_max - x) / (a_max - a_min) for x in array]
  return a_norm

def createDiffArray(metric, scenario, aim, case=None):
  # Get the modelled and observed data
  modelled = getModelledArray(metric, scenario, aim, case)
  observed = getObservedArray(metric, scenario, aim, case)
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
        theArray[row,col][zone][0] = modelled[obsTimeSteps[row]-1,col][1][zone][0] - observed[row,0][1][zone][0]  
  return theArray

def saveArray(array,name):
  fileName = os.path.join(arrayFolder, str(name))
  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')
  # Save the data  
  np.save(fileName, array)
  
@njit(parallel=True)
def getValidCells(u_array):
  # Get the number of valid cells in the map 
  cells = len(u_array)
  nanCells = float(np.sum(np.isnan(u_array)))
  validCells = cells - nanCells
  return validCells

@njit(parallel=True)
def K_calculations(row,col,obs0,obs1,mod0,mod1,cArray,allocationArray,kappaArray,validCells):
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
  cArray[0,0] = np.sum(allStates == 1)/validCells
  cArray[1,0] = np.sum(allStates == 2)/validCells
  cArray[0,1] = np.sum(allStates == 3)/validCells
  cArray[1,1] = np.sum(allStates == 4)/validCells
  cArray[2,0] = (np.sum(allStates == 1)+np.sum(allStates == 2))/validCells
  cArray[2,1] = (np.sum(allStates == 3)+np.sum(allStates == 4))/validCells
  cArray[0,2] = (np.sum(allStates == 1)+np.sum(allStates == 3))/validCells
  cArray[1,2] = (np.sum(allStates == 2)+np.sum(allStates == 4))/validCells
  cArray[2,2] = 1

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
  K_standard = (C-E)/(1-E) # (11)
  #print('year: ', str(row[1]),', parameter set: ', str(col), ', Kappa standard:',K_standard)
  
  # Save Kapa in array
  kappaArray[row[0],col] = K_standard
  # Save errors
  allocationArray[row[0],col] = A
  
def calculateKappa(scenario=None, aim=None):
  # Create arrays to store the metrices by Pontius (allocation disagreement)
  allocationArray = np.zeros((len(obsTimeSteps),numberOfIterations))
  # Create array to store Kappa
  kappaArray = np.zeros((len(obsTimeSteps),numberOfIterations)) 
  # Load data:
  urbObs = np.array(getObservedArray('urb'))
  # Loop observed years. For each year, calculate Kappa between observed map and modelled map:
  for row in enumerate(obsTimeSteps):
    # Get the number of valid cells in the map 
    validCells = getValidCells(urbObs[row[0],0,1])
    # Load modelled urban for the given obs year
    urbMod = getSubsetArray('urb',row[1])
    # Define conditions to compare simulated and actual (observed) maps
    obs1 = (urbObs[row[0],0,1] == 1)
    obs0 = (urbObs[row[0],0,1] == 0)
    
    # Loop parameter sets:
    for col in range(0,numberOfIterations):
      # Create contingency table, full of zeros (to calculate Kappa)
      cArray = np.zeros((3,3))
        
      # For year 1990 (first observation time step) there is a perfect agreement:
      if row[0] == 0:
        kappaArray[row[0],col] = 1.0
        
      else:
        # Define conditions to compare simulated and actual (observed) maps
        mod1 = (urbMod[0,col,1] == 1)
        mod0 = (urbMod[0,col,1] == 0)

        K_calculations(row,col,obs0,obs1,mod0,mod1,cArray,allocationArray,kappaArray,validCells)
        
  # Save all the metrices
  saveArray(kappaArray,'kappa')
  saveArray(allocationArray,'allocation_disagreement')
  print('Kappa and Pontius metrices calculated and saved as npy files')
  
######################################################
# KAPPA SIMULATION                                   #
######################################################
def calculateKappaSimulation(scenario=None, aim=None): 
  # Create array to store Kappa Simulation (by Van Vliet):
  kappaSimulation = np.zeros((len(obsTimeSteps),numberOfIterations))
  # Load data:
  urbObs = getObservedArray('urb')  
  # Get the number of not NaN cells
  validCells = getValidCells(urbObs[0,0,1])
  # Select the original map
  originals = getOriginal(urbObs[0,0,1], validCells)
  #original_0,original_1,p_org0,p_org1 = getOriginal(urbObs[0,0,1], validCells)
  # Loop observed years:
  for row in enumerate(obsTimeSteps):
    if row == 3:
      #original_0,original_1,p_org0,p_org1 = getOriginal(urbObs[2,0,1], validCells)
      originals = getOriginal(urbObs[2,0,1], validCells)
    # Load modelled urban for the given obs year
    urbMod = getSubsetArray('urb',row[1])
    # Define conditions to compare simulated and actual (observed) maps
    obs1 = (urbObs[row[0],0,1] == 1)
    obs0 = (urbObs[row[0],0,1] == 0)
    # Get transitions from the original map
    transitions = getTransitions(obs0,obs1,originals[0],originals[1],validCells)
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
        PO = getPO(obs0,obs1,mod0,mod1,validCells)
        # Calculate PE Transition (Van Vliet, 2013, Eq. 2.7):
        PE_transition = getPE_transition(mod0,mod1,originals,validCells,transitions)
        # Calculate Kappa Simulation (Van Vliet, 2013)
        Ks_values(row,col,kappaSimulation,PO,PE_transition)       
  # Save the data  
  saveArray(kappaSimulation,'kappa_simulation')

@njit(parallel=True)
def getOriginal(urb_array, validCells):
  # Define conditions (urban and non-urban cells) in the original map (1990 or 2006):
  original_0 = (urb_array == 0) #non-urban
  original_1 = (urb_array == 1) #urban
  p_org0 = float(np.sum(original_0)/validCells)
  p_org1 = float(np.sum(original_1)/validCells)
  return original_0,original_1,p_org0,p_org1

@njit(parallel=True)
def getTransitions(obs0,obs1,original_0,original_1,validCells):
  # Define conditions and calculate the transitions:
  act_0_0 = np.where(obs0 & original_0,1,0)
  act_0_1 = np.where(obs1 & original_0,1,0)
  act_1_0 = np.where(obs0 & original_1,1,0)
  act_1_1 = np.where(obs1 & original_1,1,0)
  p_act_0_0 = float(np.sum(act_0_0)/validCells)
  p_act_0_1 = float(np.sum(act_0_1)/validCells)
  p_act_1_0 = float(np.sum(act_1_0)/validCells)
  p_act_1_1 = float(np.sum(act_1_1)/validCells)
  return p_act_0_0,p_act_0_1,p_act_1_0,p_act_1_1
 
@njit(parallel=True)
def getPO(obs0,obs1,mod0,mod1,validCells):
  PO_urban = float(np.sum(obs1 & mod1)/validCells)
  PO_non_urban = float(np.sum(obs0 & mod0)/validCells)
  PO = PO_urban + PO_non_urban
  return PO

@njit(parallel=True)
def getPE_transition(mod0,mod1,originals,validCells,transitions):
  # Get proportion of LU classes in original map
  original_0,original_1,p_org0,p_org1 = originals
  # Get transitions from the original map
  p_act_0_0,p_act_0_1,p_act_1_0,p_act_1_1 = transitions
  # Define conditions to compare transitions in original and simulated maps
  sim_0_0 = np.where(mod0 & original_0,1,0)
  sim_0_1 = np.where(mod1 & original_0,1,0)
  sim_1_0 = np.where(mod0 & original_1,1,0)
  sim_1_1 = np.where(mod1 & original_1,1,0)
  # Get transitions from the simulated (modelled) map
  p_sim_0_0 = float(np.sum(sim_0_0)/validCells)
  p_sim_0_1 = float(np.sum(sim_0_1)/validCells)
  p_sim_1_0 = float(np.sum(sim_1_0)/validCells)
  p_sim_1_1 = float(np.sum(sim_1_1)/validCells)
  # Calculate PE Transition (Van Vliet, 2013, Eq. 2.7):
  PE_transition = p_org0 * (p_act_0_0 * p_sim_0_0 + p_act_0_1 * p_sim_0_1) +\
                  p_org1 * (p_act_1_0 * p_sim_1_0 + p_act_1_1 * p_sim_1_1)
  return PE_transition

@njit()
def Ks_values(row,col,kappaSimulation,PO,PE_transition): 
  # Calculate Kappa Simulation (Van Vliet, 2013, Eq. 2.9):
  K_simulation = (PO - PE_transition) / (1 - PE_transition)
  # Save Kappa Simulation in array
  kappaSimulation[row[0],col] = K_simulation

######################################################
  
def calcRMSE(metric, scenario, aim, case=None):
  diffArray = createDiffArray(metric,scenario,aim,case)
  # Get the number of parameter sets and number of zones
  pSets = diffArray.shape[1]
  zones = diffArray.shape[2]
  # Create empty array. Rows = nr of observed timesteps, columns = nr of parameter sets
  rmseArray = np.zeros((diffArray.shape[0],diffArray.shape[1]))

  # Calculate RMSE for each timestep and parameter set. Number of observations = number of zones
  for row in range(0,len(obsTimeSteps)):
    for col in range(0,pSets):
      # Create a list containing difference between the modelled and observed metric for the zones
      x = diffArray[row,col].flatten()
      # Remove nan values
      x = x[~numpy.isnan(x)]
      # Calculate RMSE for each zones
      rmseArray[row,col] = np.sqrt(np.mean(x**2))
      
  return rmseArray

def getRankedRMSE(metric,scenario,aim,case=None):
  # get the array with RMSE
  aRMSE = calcRMSE(metric, scenario, aim,case)
  # get the RMSE for the selected year
  sRMSE = getAveragedArray(aRMSE, scenario,aim)
  # order the parameter sets
  RMSEorder = sRMSE.argsort()
  # rank the parameter sets. rank == 0 indifies the best parameter set
  RMSEranks = RMSEorder.argsort()
  
  return RMSEranks

def getKappaIndex(scenario,aim,case=None):
  # get the array with kappa values for the whole study area or the selected zones only
  kappaArr = getKappaArray(scenario, aim, case)
  # get the kappa for the selected year
  aKAPPA = getAveragedArray(kappaArr,scenario,aim)
  # order the parameter sets
  KAPPAorder = aKAPPA.argsort()
  # rank the parameter sets
  KAPPAranks = KAPPAorder.argsort()
  # reverse the ranking of the parameter sets, as the higher value indicates a better result
  KAPPAranks = np.subtract(np.amax(KAPPAranks),KAPPAranks)
  
  return KAPPAranks

def getAllocationIndex(scenario,aim,case=None):
  # get the array with A values for the whole study area or the selected zones only
  allocationArr = getAllocationArray(scenario, aim, case)
  # get the A for the selected year
  aAllocation = getAveragedArray(allocationArr,scenario,aim)
  # order the parameter sets
  allocationOrder = aAllocation.argsort()
  # rank the parameter sets
  allocationRanks = allocationOrder.argsort()
  
  return allocationRanks

def getCalibratedIndeks(metric,scenario,case=None): 
  if metric in ['kappa','Ks']:
    # Get the ranked parameter sets based on Kappa
    ranked = getKappaIndex(scenario,'calibration',case)
  elif metric =='A':
    # Get the ranked parameter sets based on A
    ranked = getAllocationIndex(scenario,'calibration',case)
  else:
    # Get the ranked parameter sets based on RMSE
    ranked = getRankedRMSE(metric,scenario,'calibration',case)    
  calibratedIndex, = np.where(ranked == 0)
  
  return int(calibratedIndex)

def getRankedMultiobjective(metric,weights,scenario,aim='calibration',case=None):
  # get the array with RMSE
  aRMSE = calcRMSE(metric, scenario, aim,case)
  # get the RMSE for the selected year
  sRMSE = getAveragedArray(aRMSE, scenario,aim)
  # get the normalized RMSE
  n_RMSE = getNormalizedArray(sRMSE)
  # Get the Kappa array for the given aim and scenario
  aKAPPA = getKappaArray(scenario,aim, case)
  # Get the calibration value
  sKappa = getAveragedArray(aKAPPA, scenario,aim)
  # Get the normalized Kappa
  n_Kappa = getNormalizedArray(sKappa, kappa=True)
  # Join two goal function
  n_RMSE_n_Kappa = weights[0] * np.array(n_RMSE) + weights[1] * np.array(n_Kappa) # the bigger the better
  # order the parameter sets
  n_RMSE_n_Kappa_order = n_RMSE_n_Kappa.argsort()
  # rank the parameter sets
  n_RMSE_n_Kappa_ranks = n_RMSE_n_Kappa_order.argsort()
  # reverse the ranking of the parameter sets
  n_RMSE_n_Kappa_ranks = np.subtract(np.amax(n_RMSE_n_Kappa_ranks),n_RMSE_n_Kappa_ranks)
  
  return n_RMSE_n_Kappa_ranks

def getMultiObjectiveIndex(metric, w_RMSE,w_Kappa,scenario, case=None):
  # Get the ranked parameters based on multi-objective calibration
  ranked = getRankedMultiobjective(metric,[w_RMSE,w_Kappa],scenario,'calibration',case)
  # Best index is ranked 0
  calibratedIndex, = np.where(ranked == 0)
  
  return int(calibratedIndex)

def getCalibratedParameters(calibrationScenario):
  # create an array of metric names, indexes of the parameters and the parameters' values
  pArray = np.zeros((len(metricList+['kappa']),6), dtype='object')
  parameterSets = getParameterConfigurations()
  for row,m in enumerate(metricList):
    # Get the calibrated parameter index
    calibratedIndex = getCalibratedIndeks(m,calibrationScenario)
    p = parameterSets[calibratedIndex]
    pArray[row,:] = m, calibratedIndex, p[0],p[1],p[2],p[3]
  
  # Get the parameter index with the highest Kappa 
  kappaIndex = getCalibratedIndeks('kappa',calibrationScenario)
  # Get the best parameter set
  k = parameterSets[kappaIndex]
  # Fill the array
  pArray[row+1,:] = 'kappa',kappaIndex, k[0],k[1],k[2],k[3]

  return pArray

def getTopCalibratedParameters(metric, scenario, numberOfTopPerformers,case=None):
  """
  Returns an array of size numberOfTopPerformers x 7
  Each rown contains [metric name, index, validation result, p1, p2, p3, p4] 
  """
  topArray = np.zeros((numberOfTopPerformers,7), dtype='object')
  parameterSets = getParameterConfigurations()
  if metric == 'kappa':
    # Get the ranked parameter sets based on Kappa
    ranked = getKappaIndex(scenario,'calibration',case)
    # Get the validation error
    error = getKappaArray(scenario,'validation',case)
  else:
    # Get the ranked parameter sets based on RMSE
    ranked = getRankedRMSE(metric,scenario,'calibration',case)
    # Get the validation error
    error = calcRMSE(metric,scenario,'validation',case)
  
  for i in range(0,numberOfTopPerformers):
    # Get the parameter index depending on the rank i
    calibratedIndex, = np.where(ranked == i)
    sError = getAveragedArray(error, scenario, 'validation')
    p = parameterSets[calibratedIndex]
    topArray[i,[0,1,2]] = metric, calibratedIndex[0], sError[int(calibratedIndex)]
    for j in [3,4,5,6]:
      topArray[i, j] = p[0][j-3]
  return topArray

def getTopCalibratedParameters_multiobjective(metric, weights, scenario, numberOfTopPerformers,case=None):
  """
  Returns an array of size numberOfTopPerformers x 8
  Each rown contains [metric name, index, validation result RMSE, validation result Kappa, p1, p2, p3, p4]
  weights = [w_RMSE, w_Kappa] # sum=1
  """
  topArray = np.zeros((numberOfTopPerformers,8), dtype='object')
  parameterSets = getParameterConfigurations()
  # Get the ranked parameter sets based on RMSE
  multi_ranked = getRankedMultiobjective(metric,weights,scenario,aim='calibration',case=case)
  # Get the validation RMSE error
  RMSE = calcRMSE(metric,scenario,'validation',case)
  # Get the validation kappa metric
  Kappa = getKappaArray(scenario,'validation',case)
  
  for i in range(0,numberOfTopPerformers):
    # Get the parameter index depending on the rank i
    calibratedIndex, = np.where(multi_ranked == i)
    avRMSE = getAveragedArray(RMSE, scenario, 'validation')
    avKappa = getAveragedArray(RMSE, scenario, 'validation')
    p = parameterSets[calibratedIndex]
    topArray[i,[0,1,2,3]] = metric,calibratedIndex[0],avRMSE[int(calibratedIndex)],avKappa[int(calibratedIndex)]
                                                         
    for j in [4,5,6,7]:
      topArray[i, j] = p[0][j-4]
  return topArray

def getResultsEverySet(aim):
  ''' Create a dictionairy: results[country][scenario]
      Dict stores an array with shape: number of parameter sets x number of metrics
      Array stores calibration/validation values for each set for both scenarios '''
  
  # Create array to store calibration/validation results for all metrics
  results = {}
  # Get the number of parameter sets
  parameters=range(0,numberOfIterations)
    
  for c in case_studies:
    results[c]={}
    for s in scenarios:
      results[c][s] = np.empty([numberOfIterations,len(all_metrices)])   
  
  # Loop all the countries:
  for country in case_studies:
    # Get data
    resultFolder = os.path.join(os.getcwd(),'results',country, 'metrics')

    i=0
    # Loop all the metrics:
    for m in all_metrices:
      # Loop calibration scenarios:
      for scenario in scenarios:
        # Load data
        if m == 'kappa':
          an_array = getKappaArray(scenario,aim,case=country)
        elif m == 'Ks':
          an_array = getKappaSimulationArray(scenario,aim,case=country)
        elif m == 'A':
          an_array = getAllocationArray(scenario,aim,case=country)
        else:
          an_array = calcRMSE(m,scenario,aim,case=country)

        # get the calibration value of the metric for every set
        av_array = getAveragedArray(an_array,scenario,aim)
        # Save results inside the array
        results[country][scenario][:,i] = av_array
      i = i+1
      
  return results 
  
def getValidationResults():
  """
  Create a 2D array matrix:
  ## shape validation metrics x 5 goal functions * cases * 2 scenarios
  ## each cell containes RMSE or Kappa Standard, Kappa Simulation or Allocation Disagreement value
  ## this value represent the value of the validation metric (row), obtained using the given goal function,
  ## in a given case study, for a given scenario (column).
  """
  goal_functions = all_metrices
  validation_metrices = all_metrices
  # validation on 3 case studies and 2 scenarios
  validationResults = np.empty((len(validation_metrices),
                                len(goal_functions)*len(case_studies)*len(scenarios)))
  # Lop validation metrices, to get their values:
  for i,v_m in enumerate(validation_metrices):
    j=0
    for country in case_studies:
      for scenario in scenarios:
        for m in goal_functions:
          # Get the index of the calibrated parameter set
          index = getCalibratedIndeks(m, scenario,country)
          
          # an_array is an array of arrays (contains rmse/kappa arrays for every p set)
          if v_m == 'kappa':
            an_array = getKappaArray(scenario,'validation',case=country)
          elif v_m == 'Ks':
            an_array = getKappaSimulationArray('validation',case=country)
          elif v_m == 'A':
            an_array = getAllocationArray('validation',case=country)
          else:
            # Get the error for the validation metric
            an_array = calcRMSE(v_m,scenario,'validation',case=country)
            
          # Get the values for the validation period
          av_results = getAveragedArray(an_array,scenario,'validation')
          
          validationResults[i,j] = av_results[index]
          
          j+=1
          
  return validationResults

def getValidationResults_multiobjective(weights):
  """
  Create an array matrix:
  ## shape 7x24 (7 validation metrics x 4 goal functions (metrics determining parameter set) * 3 cases * 2 scenarios)
  ## each cell containes RMSEs, Kappa Standard, Kappa Simulation or Allocation Disagreement value
  ## this value represent the value of the validation metric (row),
  ## obtained using the given multiobjective goal function, combinaning patch and cell-based goal function,
  ## in a given case study, for a given scenario (column).
  """
  goal_functions = metricList
  validation_metrices = all_metrices + ['Ks','A']
  # validation on 3 case studies and 2 scenarios
  validationResults = np.empty((len(validation_metrices),len(goal_functions)*3*2))
  # Lop validation metrices, to get their values:
  for i,v_m in enumerate(validation_metrices):
    j=0
    for m in goal_functions:
      for country in case_studies:
        for scenario in scenarios:
          # Get the index of the calibrated parameter set
          index = getMultiObjectiveIndex(m, weights[0],weights[1],scenario,case=country)
          
          # an_array is an array of arrays (contains rmse/kappa arrays for every p set)
          if v_m == 'kappa':
            an_array = getKappaArray(scenario,'validation',case=country)
          elif v_m == 'Ks':
            an_array = getKappaSimulationArray('validation',case=country)
          elif v_m == 'A':
            an_array = getAllocationArray('validation',case=country)
          else:
            # Get the error for the validation metric
            an_array = calcRMSE(v_m,scenario,'validation',case=country)
          # Get the values for the validation period
          av_results = getAveragedArray(an_array,scenario,'validation')
          validationResults[i,j] = av_results[index]
          j+=1
  return validationResults

######################
#### NON DOMINATED ###
######################

def get_ND_input(scenario,aim,case):
  ''' Get the calibration'validation values for each set '''
  results = {
    'calibration':getResultsEverySet('calibration'),
    'validation':getResultsEverySet('validation')}
  # ...And get them for a selected case
  v = results[aim][case][scenario]

  return v

def get_ND_mask(case, scenario):
  """
  Create a boolean mask of the pareto-efficient points
  :param costs: An (n_points, n_costs) array
  :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
  """
  # Get the metric values to find the non dominated solutions
  costs = get_ND_input(scenario,'calibration',case)
  # Create a boolean array to select nondominated solutions
  is_efficient = np.ones(costs.shape[0], dtype = bool)
  for i, c in enumerate(costs):
    if is_efficient[i]:
      # Keep any point with a lower cost
      is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  
      is_efficient[i] = True  # And keep self
  return is_efficient

def get_ND(case, scenario, aim):
  """Filter an array and leave the non-dominated rows only"""
  # Get the calibration or validation values array
  in_results = get_ND_input(scenario,aim,case)
  # Get the non-dominated mask
  non_dominated = get_ND_mask(case, scenario)

  return in_results[non_dominated]
  
def get_ND_indices(case, scenario):
  '''Returns an array with the indices of the all non-dominated solutions'''
  v_c = get_ND_input(scenario,'calibration',case)
  # Get the mask of the non dominated solutions
  c_mask = get_ND_mask(case, scenario)
  # Get the list of the indices of the non dominated solutions
  nd_indices = np.array(range(len(v_c)))[c_mask]

  return nd_indices

def getWeights(solutions=None):
  ''' Returns an array with weights for all solutions or selected solutions only '''
  # Get the paramter sets weights
  parameterSets = getParameterConfigurations()
  # Selec indices if no specific points are requested
  if solutions is None:
    solutions = range(len(parameterSets))
  # Get the drivers
  drivers = parameters.getSuitFactorDict()[1]
  # Create an array to store the weights
  weights = np.zeros((len(drivers), len(solutions)))
  # Fill the data for each point
  for i, col in enumerate(weights.T):
    if solutions == range(len(parameterSets)):
      weights[:,i] = parameterSets[solutions[i]]
    else:
      weights[:,i] = parameterSets[solutions[i]][0]
 
  return weights

def getAverageWeights(case, scenario, indices=None):
  ''' Returns the average importance (weight) of a given urban growth driver
      for all solutions or selected solutions only '''
  # Get the drivers
  drivers = parameters.getSuitFactorDict()[1]
  # Get weights:
  weights = getWeights(indices)
  # Create a list with the average weights
  av_weights = np.average(weights, axis=1)

  return av_weights

def get_ND_n_masks(case, scenario, trade_off = False, eps=0.9):
  """
  Create a boolean mask of the selected n solutions selected from the non dominated points 
  :param nd_results: An (n_points, costs) array of non-dominated solutions
  :return:
    a mask for n points for of each objective
    points in order of objectives
  """
  # Get parameters
  #parameterSets = getParameterConfigurations()
  # Get the non-dominated mask
  #nd_mask = get_ND_mask(case, scenario)
  # Get the non dominated parameter sets only
  #nd_parameters = parameterSets[nd_mask]
  # Get the results values
  in_results = get_ND(case, scenario, aim='calibration')
  # Create an array to store condition boolean arrays
  c_list =[]
  # Save a boolean list
  for i, col in enumerate(in_results.T):
    c_list.append(in_results[:,i] == in_results.min(axis=0)[i])

  # Change into an array 
  c_array = np.array(c_list)
  
  '''# Create a mask to find the values selected in terms of n objectives
  c_n_mask = [ any(col) for col in c_array.T ]

  ## Get the last point.
  # The ideal point is in the 0 point of axes
  s_ideal = np.array([0,0,0])
  # Get the normalized values of non-dominated solutions in order to compare them
  r_nd_c_norm = (in_results.max(axis=0) - in_results) / (in_results.max(axis=0) - in_results.min(axis=0))
  # Get the difference between the non-dominated points and the "ideal" point
  r_nd_c_s_mo = np.array([np.linalg.norm(r - s_ideal) for r in r_nd_c_norm])
  # Rank the distances from the smallest to the biggest. First create a temp array
  temp = r_nd_c_s_mo.argsort()
  # Create ranks
  ranks = numpy.empty_like(temp)
  # Assign ranks
  ranks[temp] = numpy.arange(len(r_nd_c_s_mo))
  # If we're looking for the realistic solutions, include the rationality
  if trade_off is True:
    rational = False
    i=0
    # Loop to find the rational one
    while rational is False:
      # Create a boolean mask selecting the point with the minimum distance to the "ideal" point
      c_mo = ranks==i
      # Select the point
      a_p_set = nd_parameters[c_mo]
      # If any weight crosses the epsilon, look for the seond closest point:
      if np.any(a_p_set[0]>eps):
        i+=1
      else:
        rational = True
  else:
    c_mo = ranks==0

  # Append to the list
  c_list.append(c_mo)
  # Change into an array
  c_array = np.array(c_list)'''

  return c_array

def get_ND_n_indices(case, scenario):
  # Get non dominated indices
  indices_nd = get_ND_indices(case, scenario)
  # Get the n solutions mask array
  n_mask_array = get_ND_n_masks(case, scenario)
  # Subset indices of the n solutions, keeping the order of n objectives
  n_indices = [indices_nd[condition] for condition in n_mask_array]
  
  return n_indices

def get_ND_1_mask(case, scenario, solution_space):
  ''' Return a mask of the 1 solutions based on the all objectives
      solution_space in ['all', 'nondominated'] '''
  # Get the indices of the selected solutions in terms of the n objecitves
  nd_n_indices = get_ND_n_indices(case, scenario)
  # Get the average value of the weights
  n_weights_av = getAverageWeights(case, scenario, nd_n_indices)
  # Get the weights for every possible paramter set
  weights = {
    'all': getWeights(),
    'nondominated': getWeights(get_ND_indices(case, scenario)) }
  # Calculate the distance to the averaged weight for every possible set
  distance_to_averaged = np.array(
    [distance.euclidean(n_weights_av, col) for col in weights[solution_space].T])
  # Find the closest point
  theMask = distance_to_averaged==distance_to_averaged.min()

  return theMask

def get_ND_1_index(case, scenario, solution_space):
  ''' Returns index of the 1 solution based on the all objectives
      solution_space in ['all', 'nondominated'] <- point to be selected from'''
  # Get the indices from the solutions space
  indices_space = {
    'all': range(len(getParameterConfigurations())),
    'nondominated': get_ND_indices(case, scenario) }
  # Get the mask with the solution
  aMask = get_ND_1_mask(case, scenario, solution_space)
  # Get the inbdex
  nd_1_index = np.array(indices_space[solution_space])[aMask]

  return nd_1_index

def get_ND_n_1_indices(case, scenario, solution_space):
  ''' Returns an array with the n+1 indices for n objectives from the non-dominated solutions
      solution_space in ['all', 'nondominated'] <- point to be selected from'''
  # Get the indices of the n objectives
  nd_n_indices = get_ND_n_indices(case, scenario)
  # Get the index of the multiobjective solution
  nd_1_index = get_ND_1_index(case, scenario, solution_space)
  # Get the array of masks of the selected points
  nd_n_1_indices = np.append(nd_n_indices, nd_1_index)
  
  return nd_n_1_indices

def get_ND_n_1_masks(case, scenario, solution_space):
  ''' Create n+1 boolean masks of the selected n+1 solutions selected from the non dominated points 
      solution_space in ['all', 'nondominated'] '''
  # Get all indices possible
  indices_all = np.arange(len(getParameterConfigurations()))
  # Get the indices of n+1 objectives
  indices_n_1 = get_ND_n_1_indices(case, scenario, solution_space)
  # Create an array of masks
  masks = np.array([aidx == indices_all for aidx in indices_n_1])
  # Create one mask
  mask = np.array([ np.any(col) for col in masks.T ])
  # Subset non dominated solutions only
  c_array = mask[get_ND_mask(case, scenario)]

  return c_array

def get_ND_n_1(case, scenario, aim, solution_space):
  ''' Returns n+1 points from the results, keeping their order'''
  # Get results depending on aim
  results_nd = get_ND_input(scenario,aim,case)
  # Get the indices of the n+1 solutions
  idx_n_1 = get_ND_n_1_indices(case, scenario, solution_space)
  # Subset points
  results_nd_n_1 = np.array([results_nd[i] for i in idx_n_1])

  return results_nd_n_1

"""


def get_ND_n_1_mask(case, scenario, trade_off = None):
  ''' Flattens the n+1 simension mask into 1D mask. Order of the solutions is lost '''
  # Get the conditions
  c_array = get_ND_n_1_masks(case, scenario, trade_off)
  # Create a mask to find the values selected in terms of four objectives
  c_mask = [any(col) for col in c_array.T]
  return c_mask

  
def get_ND_n_1(case, scenario, aim, solution_space, trade_off = None):
  ''' Create an array of the selected n+1 points in order of objectives '''
  # Get the results values
  out_results = get_ND(case, scenario, aim)
  # Get the conditions from input array
  c_array = get_ND_n_1_masks(case, scenario, solution_space)
  # Create a list to store points
  p_list =[]
  # Save a boolean list
  for i in range(len(c_array)):
    p_list.append(out_results[c_array[i]].flatten())
  # Change into an array
  p_array = np.array(p_list)
  
  return p_array
"""


