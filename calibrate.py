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
all_metrices = metricList+['kappa']
case_studies = ['IE','IT','PL']

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
arrayFolder = os.path.join(os.getcwd(),'results',country,'metrics')


#################
### FUNCTIONS ###
#################

def getModelledArray(metric,scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(os.getcwd(),'results',case,'metrics')
  return np.load(os.path.join(folder, metric + '.npy'))

def getObservedArray(metric,scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(os.getcwd(),'results',case,'metrics')
  return np.load(os.path.join(folder, metric + '_obs.npy'))
  
def getParameterConfigurations(case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(os.getcwd(),'results',case,'metrics')
  return np.load(os.path.join(folder, 'parameter_sets.npy'))

def getKappaArray(scenario=None,aim=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(os.getcwd(),'results',case,'metrics')
  else:  
    return np.load(os.path.join(folder, 'kappa.npy'))

def getKappaSimulationArray(aim=None,scenario=None, case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(os.getcwd(),'results',case,'metrics')
  else:  
    return np.load(os.path.join(folder, 'kappa_simulation.npy'))

def getAllocationArray(aim=None,scenario=None,case=None):
  if case is None:
    folder = arrayFolder
  else:
    folder = os.path.join(os.getcwd(),'results',case,'metrics')
  else:  
    return np.load(os.path.join(folder, 'allocation_disagreement.npy'))

def getSelectedArray(array, scenario, aim):
  # end_year is a list containing index of the year, which is the end of calibration/validation period,
  # defined by the scenario
  end_year = parameters.getCalibrationPeriod()[scenario][aim]
  return array[end_year]

def getNormalizedArray(array, kappa=None):
  a_max = np.amax(array)
  a_min = np.amin(array)
  if kappa==True: 
    a_norm = [(x - a_min) / (a_max - a_min) for x in array]
  else:
    a_norm = [(a_max - x) / (a_max - a_min) for x in array]
  return a_norm

def createDiffArray(metric, scenario, aim, case=None):
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

def calculateKappa(scenario=None, aim=None):
  # Create arrays to store the metrices by Pontius (quantity disagreement and allocation disagreement)
  quantityArray = np.zeros((len(obsTimeSteps),numberOfIterations))
  allocationArray = np.zeros((len(obsTimeSteps),numberOfIterations))
  # Create array to store transition states:
  transArray = np.zeros((len(obsTimeSteps),numberOfIterations,3,3))
  # Create array to store Kappa
  kappaArray = np.zeros((len(obsTimeSteps),numberOfIterations))
    
  # Load data:
  urbObs = getObservedArray('urb')
  cells = len(urbObs[0,0,1])
  nanCells = sum(np.isnan(urbObs[0,0,1]))
  validCells = cells - nanCells
      
  # Loop observed years. For each year, calculate Kappa between observed map and modelled map:
  for row in enumerate(obsTimeSteps):
    print(row)
    # Load modelled urban for the given obs year
    urbMod = np.load(os.path.join(arrayFolder, 'urb_subset_'+str(row[1]) + '.npy'))
    if scenario==3:
      urbMod = subsetUrbanZones(urbMod,aim)

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
    fileName = os.path.join(arrayFolder, str(name))

    # Clear the directory if needed
    if os.path.exists(fileName + '.npy'):
        os.remove(fileName + '.npy')

    # Save the data  
    np.save(fileName, metrices[name])
    
  print('Kappa and Pontius metrices calculated and saved as npy files')

def calculateKappaSimulation(scenario=None, aim=None):
  # Create array to store Kappa Simulation (by Van Vliet):
  kappaSimulation = np.zeros((len(obsTimeSteps),numberOfIterations))
    
  # Load data:
  urbObs = getObservedArray('urb')  
  cells = len(urbObs[0,0,1])
  nanCells = sum(np.isnan(urbObs[0,0,1]))
  validCells = cells - nanCells

  # Define conditions (urban and non-urban cells) in the original map (1990):
  original_0 = (urbObs[0,0,1] == 0) #non-urban
  original_1 = (urbObs[0,0,1] == 1) #urban
  p_org0 = sum(original_0)/validCells
  p_org1 = sum(original_1)/validCells
      
  # Loop observed years:
  for row in enumerate(obsTimeSteps):
    # Load modelled urban for the given obs year
    urbMod = np.load(os.path.join(arrayFolder, 'urb_subset_'+str(row[1]) + '.npy'))

    # Define conditions to compare simulated and actual (observed) maps
    obs1 = (urbObs[row[0],0,1] == 1)
    obs0 = (urbObs[row[0],0,1] == 0)
    # Define conditions and calculate the transitions:
    act_0_0 = np.where(obs0 & original_0,1,0)
    act_0_1 = np.where(obs1 & original_0,1,0)
    act_1_0 = np.where(obs0 & original_1,1,0)
    act_1_1 = np.where(obs1 & original_1,1,0)
    p_act_0_0 = sum(act_0_0)/validCells
    p_act_0_1 = sum(act_0_1)/validCells
    p_act_1_0 = sum(act_1_0)/validCells
    p_act_1_1 = sum(act_1_1)/validCells
    
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
        sim_0_0 = np.where(mod0 & original_0,1,0)
        sim_0_1 = np.where(mod1 & original_0,1,0)
        sim_1_0 = np.where(mod0 & original_1,1,0)
        sim_1_1 = np.where(mod1 & original_1,1,0)
        
        # Calculate PE Transition (Van Vliet, 2013, Eq. 2.7):
        p_sim_0_0 = sum(sim_0_0)/validCells
        p_sim_0_1 = sum(sim_0_1)/validCells
        p_sim_1_0 = sum(sim_1_0)/validCells
        p_sim_1_1 = sum(sim_1_1)/validCells
        
        PE_transition = p_org0 * (p_act_0_0 * p_sim_0_0 + p_act_0_1 * p_sim_0_1) +\
                        p_org1 * (p_act_1_0 * p_sim_1_0 + p_act_1_1 * p_sim_1_1)

        # Calculate Kappa Simulation (Van Vliet, 2013, Eq. 2.9):
        K_simulation = (PO - PE_transition) / (1 - PE_transition)
        
        print('year: ', str(row[1]),', parameter set: ', str(col), ', Kappa simulation:',K_simulation)
        
        # Save Kappa Simulation in array
        kappaSimulation[row[0],col] = K_simulation
      
  # Set the name of the file
  fileName = os.path.join(arrayFolder, str('kappa_simulation'))
  if scenario==3:
    fileName = os.path.join(arrayFolder, 'kappa_simulation_'+str(aim[:3]))
      
  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, kappaSimulation)

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
  sRMSE = getSelectedArray(aRMSE, scenario,aim)
  # order the parameter sets
  RMSEorder = sRMSE.argsort()
  # rank the parameter sets. rank == 0 indifies the best parameter set
  RMSEranks = RMSEorder.argsort()
  
  return RMSEranks

def getKappaIndex(scenario,aim,case=None):
  # get the array with kappa values for the whole study area or the selected zones only
  kappaArr = getKappaArray(scenario, aim, case)
  # get the kappa for the selected year
  aKAPPA = getSelectedArray(kappaArr,scenario,aim)
  # order the parameter sets
  KAPPAorder = aKAPPA.argsort()
  # rank the parameter sets
  KAPPAranks = KAPPAorder.argsort()
  # reverse the ranking of the parameter sets, as the higher value indicates a better result
  KAPPAranks = np.subtract(np.amax(KAPPAranks),KAPPAranks)
  
  return KAPPAranks

def getCalibratedIndeks(metric,scenario,case=None): 
  if metric == 'kappa':
    # Get the ranked parameter sets based on Kappa
    ranked = getKappaIndex(scenario,'calibration',case)
  else:
    # Get the ranked parameter sets based on RMSE
    ranked = getRankedRMSE(metric,scenario,'calibration',case)    
  calibratedIndex, = np.where(ranked == 0)
  
  return int(calibratedIndex)

def getRankedMultiobjective(metric,weights,scenario,aim='calibration',case=None):
  # get the array with RMSE
  aRMSE = calcRMSE(metric, scenario, aim,case)
  # get the RMSE for the selected year
  sRMSE = getSelectedArray(aRMSE, scenario,aim)
  # get the normalized RMSE
  n_RMSE = getNormalizedArray(sRMSE)
  # Get the Kappa array for the given aim and scenario
  aKAPPA = getKappaArray(scenario,aim, case)
  # Get the calibration value
  sKappa = getSelectedArray(aKAPPA, scenario,aim)
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

def saveResults(array, scenario, fileName):
  with open(os.path.join(arrayFolder,'Scenario'+'_'+str(scenario)+'_'+fileName),
            'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for row in array:
      writer.writerow(row)

def getLog():
  log = [
    ['country: ',parameters.getCountryName()],
    ['observed time steps: ', parameters.getObsTimesteps()],
    ['parameters (min, max, step): ',parameters.getParametersforCalibration()],
    ['alpha: ',parameters.getAlphaValue()]]
  return log

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
    sError = getSelectedArray(error, scenario, 'validation')
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
    avRMSE = getSelectedArray(RMSE, scenario, 'validation')
    avKappa = getSelectedArray(RMSE, scenario, 'validation')
    p = parameterSets[calibratedIndex]
    topArray[i,[0,1,2,3]] = metric,calibratedIndex[0],avRMSE[int(calibratedIndex)],avKappa[int(calibratedIndex)]
                                                         
    for j in [4,5,6,7]:
      topArray[i, j] = p[0][j-4]
  return topArray

def getValidationResults():
  """
  Create a 2D array matrix:
  ## shape 7x30 (7 validation metrics x 5 goal functions (metrics determining parameter set) * 3 cases * 2 scenarios)
  ## each cell containes RMSE or Kappa Standard, Kappa Simulation or Allocation Disagreement value
  ## this value represent the value of the validation metric (row), obtained using the given goal function,
  ## in a given case study, for a given scenario (column).
  """
  goal_functions = all_metrices
  validation_metrices = all_metrices + ['Ks','A']
  # validation on 3 case studies and 2 scenarios
  validationResults = np.empty((len(validation_metrices),len(goal_functions)*3*2))
  # Lop validation metrices, to get their values:
  for i,v_m in enumerate(validation_metrices):
    j=0
    for m in goal_functions:
      for country in case_studies:
        for scenario in [1,2]:
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
          av_results = getSelectedArray(an_array,scenario,'validation')
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
        for scenario in [1,2]:
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
          av_results = getSelectedArray(an_array,scenario,'validation')
          validationResults[i,j] = av_results[index]
          j+=1
  return validationResults 

###############################################################################
############################ CALBRATE AND VALIDATE ############################
###############################################################################

def calibrate_validate(scenario):
  # Get parameter values
  p = getCalibratedParameters(scenario)
  # Get the parameter sets' indices only
  indexes = np.array(p)[:,1]

  c={}
  for aim in ['calibration', 'validation']:
    kappaArray = getKappaArray(scenario,aim)
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
          rmseArray = calcRMSE(metricCol,scenario,aim)
          v[i,j] = np.mean(rmseArray[aim_period, int(index)])
    c[aim] = v

  # Create a list containing all errors
  errors = [['Calibration:']]+list(c['calibration'])+[['Validation:']]+list(c['validation'])
  # Save the errors
  saveResults(['Scenario '+str(scenario)]+getLog()+list(p)+errors, scenario, 'calibration_validation.csv')

def multiobjective(scenario):
  # Define weights for the multi-objective calibration. Weights are assigned to RMSE and kappa values.
  # function arguents: weight_min,weight_max+1,weight_step
  weight_min = 0
  weight_max = 10
  weight_step = 1
  weights = range(weight_min,weight_max+weight_step,weight_step)
  # Create array to store multiobjective validation results. The array stores indices and validation results:
  result = np.empty((len(metricList),1),dtype='object')
  # 4 rows: index / RMSE / Kappa / goal function for each metric
  # Columns: weight combinations (e.g. 0: [RMSE_weight = weight_min, Kappa_weight = weight_max])
  ncols = int((weight_max-weight_min)/weight_step)+1
  # get Kappa for validation before the loop
  kappa = getKappaArray(scenario, 'validation')
  # Get kappa values for selected period
  avKappa = getSelectedArray(kappa, scenario, 'validation')
  # Normalize kappa array
  norKappa = getNormalizedArray(avKappa, kappa=True)
 
  # Calibrate goal function for each metric          
  for row,metric in enumerate(metricList):
    multiobjective = np.zeros((4,ncols))
    for w in weights:
      # Assign weights
      rmse_weight = w/10
      kappa_weight = 1 - rmse_weight    
      # Get the index of the calibrated parameter set based on the multi-objective goal function
      multiIndex = getMultiObjectiveIndex(metric,rmse_weight,kappa_weight,scenario,'calibration')
      # Validate
      # Get the RMSE array
      rmse = calcRMSE(metric, scenario,'validation')
      # Get RMSE values for selected period
      avRMSE = getSelectedArray(rmse, scenario, 'validation')
      # Normalize RMSE array
      norRMSE = getNormalizedArray(avRMSE)
      # Save selected index, the average RMSE and the average Kappa
      multiobjective[0, w] = multiIndex
      multiobjective[1, w] = avRMSE[multiIndex]
      multiobjective[2, w] = avKappa[multiIndex]
      # Results of the goal function for the validation data
      multiobjective[3, w] = rmse_weight * norRMSE[multiIndex] + kappa_weight * norKappa[multiIndex]
    result[row,0] = multiobjective

  # Save results
  saveResults(result, scenario, 'multiobjective.csv')

  # Save results as npy
  # Set the name of the file
  fileName = os.path.join(arrayFolder, 'scenario_'+str(scenario)+'_multiobjective')
  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')
  # Save the data  
  np.save(fileName, result)
    
  return result

  

