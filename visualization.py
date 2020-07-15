# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import metrics
import numpy as np
import parameters
import calibrate
#from pcraster.framework import *
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import scipy.stats as stats
from openpyxl import load_workbook

#### Global variables:
# Get metrics
metricNames = parameters.getSumStats()
all_metrices = metricNames+['kappa']
case_studies = ['IE','IT','PL']

# Get number of zones and parameters
numberOfZones = parameters.getNumberOfZones()
numberOfParameters = parameters.getNumberofIterations() # use this one only for colors

# Create colors for zones and parameters
zoneColors = plt.cm.rainbow(np.linspace(0,1,numberOfZones))
parameterColors = plt.cm.rainbow(np.linspace(0,1,numberOfParameters))
countryColors = {'IE':'mediumspringgreen','IT':plt.cm.rainbow(np.linspace(0,1,3))[0],'PL':plt.cm.rainbow(np.linspace(0,1,3))[2]}
functionColors = plt.cm.rainbow(np.linspace(0,1,4))
c_obs = ['crimson', (0.97,0.97,0.94),'black']
c_mod = ['lavender', 'dimgrey', 'gold']
driverColors = ['royalblue','mediumvioletred','teal','darkkhaki'] # NEIGH, TRAIN, TRAVEL, LU

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsTimeSteps = parameters.getObsTimesteps()
observedYears = [parameters.getObsYears()[y] for y in obsTimeSteps]

# Path to the folder with the metrics stored
country = parameters.getCountryName()
resultFolder = os.path.join(os.getcwd(),'results',country, 'metrics')

# Set plotting variables
VERY_SMALL = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=VERY_SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=VERY_SMALL)    # fontsize of the tick labels
plt.rc('legend', fontsize=VERY_SMALL)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)   # fontsize of the figure title

'''tickDict = {
  'fontsize': VERY_SMALL,
  'fontweight': 'bold'}'''

# Name the calibratio and validation metrices and functions
metric_units = ['RMSE','RMSE','RMSE','RMSE','RMSE','RMSE','RMSE','RMSE','K','Ks','A']
calibration_metrices = [ m.upper() for m in metricNames ] + ['K']
validation_metrices = calibration_metrices + ['Ks', 'A']
goal_functions = ['f('+x.upper()+')' for x in metricNames] + ['h1(K)']
validation_functions = goal_functions + ['h2(Ks)','h3(A)']
mo_goal_functions = ['q('+cm+',K)' for cm in calibration_metrices[:-1]]
# Name the possible combinations of case studies and countries:
cases = [(country,scenario) for country in case_studies for scenario in [1,2]]
#################
### FUNCTIONS ###
#################

def clearCreatePath(path, name):
  if not os.path.isdir(path):
    os.mkdir(path)
  # Create dir of a file
  wPath = os.path.join(path, name)
  if os.path.exists(wPath):
      os.remove(wPath)
  return wPath
  
def setNameClearSave(figName, scenario=None):
  # Set the name and clear the directory if needed
  if scenario is None:
    name = ''
  else:
    name = '_scenario_'+str(scenario)
  # Create figure dir
  fig_dir = os.path.join(os.getcwd(),'results','figures')
  wPath = clearCreatePath(fig_dir, figName+name)
  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,wPath), bbox_inches = "tight",dpi=300)
  plt.close('all')

def getAverageResultsArrayEverySet(aim):
  results = np.empty((len(all_metrices),6,286)) # 6=3 case studies * 2 scenarios, 165 parameter sets
  for i,m in enumerate(all_metrices):
    j=0
    for country in case_studies:
      for scenario in [1,2]:
        # an_array is an array of arrays (contains rmse/kappa arrays for every p set)
        if m == 'kappa':
          an_array = calibrate.getKappaArray(scenario,aim,case=country)
        else:
          an_array = calibrate.calcRMSE(m,scenario,aim,case=country)
        # get the goal function value of the metric for every parameter set  
        av_array = calibrate.getAveragedArray(an_array,scenario,aim)
        results[i,j] = av_array
        j+=1
  return results

#dummykappa
'''c = ['g','b','r']

for country in case_studies:
  index1 = calibrate.getCalibratedIndeks('kappa',1,case=country)
  index2 = calibrate.getCalibratedIndeks('kappa',2,case=country)
  print(index1,index2)
  an_array1 = calibrate.getKappaArray(case=country)
  an_array2 = calibrate.getKappaSimulationArray(case=country)
  plt.plot([1990,2000,2006,2012,2018],an_array1[:,index1],'-', label='Kappa '+country)
  plt.plot([1990,2000,2006,2012,2018],an_array2[:,index1],'--', label='Kappa Simulation '+country)
  plt.legend()
    
  plt.xticks([1990,2000,2006,2012,2018],[1990,2000,2006,2012,2018])
  plt.ylim([0.55,1.2])

  plt.show()'''


def createNormalizedResultDict(metrics,aim):
  # Create array to store calibration/validation results for all metrics
  results = {}
  # Get the number of parameter sets
  parameterSets = calibrate.getParameterConfigurations()
  n = len(parameterSets)
  parameters=range(0,n)
    
  for c in case_studies:
    results[c]={}
    for s in [1,2]:
      results[c][s] = np.empty([n,len(metrics)])   
  
  # Loop all the countries:
  for country in case_studies:
    # Get data
    resultFolder = os.path.join(os.getcwd(),'results',country, 'metrics')

    i=0
    # Loop all the metrics:
    for m in metrics:
      # Loop calibration scenarios:
      for scenario in [1,2]:
        # Load data
        if m == 'kappa':
          an_array = calibrate.getKappaArray(scenario,aim,case=country)
        elif m == 'Ks':
            an_array = calibrate.getKappaSimulationArray(scenario,aim,case=country)
        elif m == 'A':
          an_array = calibrate.getAllocationArray(scenario,aim,case=country)
        else:
          an_array = calibrate.calcRMSE(m,scenario,aim,case=country)

        # get the calibration value of the metric for every set
        av_array = calibrate.getAveragedArray(an_array,scenario,aim)
        if m in ['kappa','Ks']:
          k = True
        else:
          k = False
        n_array = calibrate.getNormalizedArray(av_array,kappa=k)
        results[country][scenario][:,i] = n_array
      i = i+1
      
  return results 

def is_pareto_efficient_simple(costs):
  """
  Find the pareto-efficient points
  :param costs: An (n_points, n_costs) array
  :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
  """
  is_efficient = np.ones(costs.shape[0], dtype = bool)
  for i, c in enumerate(costs):
      if is_efficient[i]:
          is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost
          is_efficient[i] = True  # And keep self
  return is_efficient

def getCountryColors():
  alpha = {1:0.9,2:0.6}
  c_color=[]
  for case in case_studies:
    for s in [1,2]:
      a_color = colors.to_rgba(countryColors[case])[:-1]+(alpha[s],)
      c_color.append(a_color)
  return c_color

def saveArrayAsCSV(array, filename, row_names, col_names):
  # Convert to dataframe
  df = pd.DataFrame(array, index = row_names, columns = col_names)
  # Create table dir
  table_dir = os.path.join(os.getcwd(),'results','stats')
  table_name = filename
  table_path = clearCreatePath(table_dir, table_name)
  # Save table
  df.to_csv(table_path, index=True, header=True, sep=' ')
  
def saveArrayAsExcel(array, filename, row_names= None, col_names= None, sheet_name = None):
  # Convert to dataframe
  df = pd.DataFrame(array, index = row_names, columns = col_names)
  # Create table dir
  table_dir = os.path.join(os.getcwd(),'results','stats')
  table_name = filename+'.xlsx'
  table_path = clearCreatePath(table_dir, table_name)
  # Save table
  df.to_excel(table_path, sheet_name=sheet_name, index=True, header=True)

def appendArrayAsExcel(df, filename, row_names= None, col_names= None, sheet_name = None):
  # Create table dir
  table_dir = os.path.join(os.getcwd(),'results','stats',filename+'.xlsx')
  # Create the writer
  book = load_workbook(table_dir)
  writer = pd.ExcelWriter(table_dir, engine = 'openpyxl')
  writer.book = book
  # Save table
  df.to_excel(writer, sheet_name=sheet_name, index=True, header=True)
  # Close the Pandas Excel writer and output the Excel file.
  writer.save()
  writer.close()

def getSingleGoalFunctionStats(gf_metric,no_top_solutions):
  """
  Get statistics for validation of best performers for single-objective goal function
  A case study with a calibration scenario is a single case here ==> 6 cases
  returns: array 6 x no_top_solutions x 11 (number of validation_metrices)
  """
  # Create an array to store the statArray (results for each case)
  proArray = np.empty((6,no_top_solutions,11))
  # Create empty array:
  statArray = np.empty((no_top_solutions,len(validation_functions)))
  # Get data for each case and scenario seperately:
  i=0
  for country, scenario in [ [c,s] for c in case_studies for s in [1,2]]:
    if gf_metric == 'K':
      # Get ranked parameter sets based on Kappa:
      ranked = calibrate.getKappaIndex(scenario,'calibration',country)
    else:
      ranked = calibrate.getRankedRMSE(gf_metric,scenario,'calibration',country)
    # Make an ampty list for top indices:
    top_indices = []
    # Get the indices of the  parameter sets for each of the top solutions:
    for n in range(no_top_solutions):
      calibratedIndex, = np.where(ranked == n)
      # Save the index
      top_indices.append(calibratedIndex)
      
    ## Get the validation metrices for the top parameter sets.
    # Loop validation metrics:
    for v, v_m in enumerate(validation_metrices):
      # Get the array of arrays (contains rmse/kappa arrays for every p set)
      if v_m == 'K':
        an_array = calibrate.getKappaArray(scenario,'validation',case=country)
      elif v_m == 'Ks':
        an_array = calibrate.getKappaSimulationArray('validation',case=country)
      elif v_m == 'A':
        an_array = calibrate.getAllocationArray('validation',case=country)
      else:
        # Get the error for the validation metric
        an_array = calibrate.calcRMSE(v_m,scenario,'validation',case=country)
      # Get the average values for the vaidation period
      av_results = calibrate.getAveragedArray(an_array,scenario,'validation')
      # Get the value for each of the indices
      av_results_top = [ av_results[i] for i in top_indices ]
      # Fill the column for the selected validation metric:
      statArray[:,v] = av_results_top

    '''## Save the array as csv file.
    table_name = 'h1(K)_'+country+str(scenario)+'_'+str(no_top_solutions)+'top_solutions.csv'
    saveArrayAsCSV(statArray, table_name, row_names=range(no_top_solutions), col_names = validation_metrices)'''

    # Update the array for the all results
    proArray[i] = statArray
    i+=1
  return proArray

def getMultiobjectiveGoalFunctionStats(no_top_solutions, weights): 
  """
  weights = [w_RMSE, w_Kappa]
  Get statistics for validation of best performers
  for multiobjective goal function combining Kappa and a landscape metric ==> 4 landscape metrics
  A case study with a calibration scenario is a single case here ==> 6 cases
  returns: array 4 x 6 x no_top_solutions x no_validation_metrices (=11)
  """
  # Create an array to store the statArray (results for each case)
  proArray = np.empty((len(metricNames),6,no_top_solutions,11))
  # Create empty array:
  statArray = np.empty((no_top_solutions,len(validation_functions)))
  # Assign metrics used for validation:
  validation_metrices = all_metrices + ['Ks','A']
  # Get data for each metric, case and scenario seperately:
  
  for m, metric in enumerate(metricNames):
    i=0
    for country in case_studies:
      for scenario in [1,2]:
        # Get ranked parameter sets based on nultiobjective goal function:
        ranked_multiobjective = calibrate.getRankedMultiobjective(metric,weights,scenario,'calibration',country)
        # Make an ampty list for top indices:
        top_indices = []
        # Get the indices of the  parameter sets for each of the top solutions:
        for n in range(no_top_solutions):
          calibratedIndex, = np.where(ranked_multiobjective == n)
          # Save the index
          top_indices.append(calibratedIndex)
        
        ## Get the validation metrices for the top parameter sets.
        # Loop validation metrics:
        for v, v_m in enumerate(validation_metrices):
          # Get the array of arrays (contains rmse/kappa arrays for every p set)
          if v_m == 'kappa':
            an_array = calibrate.getKappaArray(scenario,'validation',case=country)
          elif v_m == 'Ks':
            an_array = calibrate.getKappaSimulationArray('validation',case=country)
          elif v_m == 'A':
            an_array = calibrate.getAllocationArray('validation',case=country)
          else:
            # Get the error for the validation metric
            an_array = calibrate.calcRMSE(v_m,scenario,'validation',case=country)
          # Get the average values for the vaidation period
          av_results = calibrate.getAveragedArray(an_array,scenario,'validation')
          # Get the value for each of the indices
          av_results_top = [ av_results[i] for i in top_indices ]
          # Fill the column for the selected validation metric:
          statArray[:,v] = av_results_top

        '''## Save the array as csv file.
        table_name = 'multiobjective_goal_f_'+metric.upper()+'_and_h1(K)_'+country+str(scenario)+'_'+str(no_top_solutions)+'top_solutions.csv'
        saveArrayAsCSV(statArray, table_name, row_names=range(no_top_solutions), col_names = validation_metrices)'''

        # Update the array for the all results
        proArray[m,i] = statArray
        i+=1
  return proArray

def reshapeSingleObjectiveResults_cases_scenarios_aggregated(single_gf_metric, no_top_solutions):
  # Get the results for the top performing parameters
  results = getSingleGoalFunctionStats(single_gf_metric,no_top_solutions)
  # Join cases into one array
  rResults = np.reshape(results,(results.shape[0]*results.shape[1],results.shape[2]))
  return rResults
  
def reshapeMultiObjectiveResults_cases_scenarios_aggregated(no_top_solutions, weights):
  # Get the results for the top performing parameters
  multiResults = getMultiobjectiveGoalFunctionStats(no_top_solutions, weights)
  # Join cases into one array
  rMultiResults = np.reshape(multiResults,
                             (multiResults.shape[0],
                              multiResults.shape[1]*multiResults.shape[2], # join cases and scenarios into one array
                              multiResults.shape[3]))
  return rMultiResults

def reshapeSingleObjectiveResults_scenarios_aggregated(single_gf_metric, no_top_solutions):
  """
  Joins validation results for two scenarios
  returns: array of size: number of case studies x no of top performers x no of validation metrics
  """
  # Get the results for the top performing parameters
  results = getSingleGoalFunctionStats(single_gf_metric,no_top_solutions)
  # Join cases into one array
  rResults = np.reshape(results,(int(results.shape[0]/2),results.shape[1]*2,results.shape[2])) 
  '''# Split the the array to get list of case studies, with scenarios as subarrays
  sResults = np.array(np.split(results,len(case_studies)))
  # Join two scenarios within one case study
  rResults = np.concatenate((sResults[:,0], sResults[:,1]), axis=1)'''
  # Return array
  return rResults

def reshapeMultiObjectiveResults_scenarios_aggregated(no_top_solutions, weights):
  """
  Joins validation results for two scenarios
  returns: array of size: multiobjective goal functions x case studies x top performers x validation metrics
  """
  # Get the results for the top performing parameters
  multiResults = getMultiobjectiveGoalFunctionStats(no_top_solutions, weights)
  # Join cases into one array
  rMultiResults = np.reshape(multiResults,
                             (multiResults.shape[0],
                              int(multiResults.shape[1]/2),
                              multiResults.shape[2]*2, # join cases and scenarios into one array
                              multiResults.shape[3]))
  return rMultiResults

def saveValidationResults_to_excel(): 
  # Get validation results
  results = calibrate.getValidationResults()
  # Get the cases
  cases =[country+str(scenario) for country in case_studies for scenario in [1,2]]
  # Assign rows and columns names
  rows = validation_metrices
  cols = [gf+'_'+case for gf in goal_functions for case in cases]
  # Save as excel file
  saveArrayAsExcel(results, 'validation_results_excel', sheet_name = 'val_results',
                   row_names=rows, col_names=cols)

def saveMultiobjectiveValidationResults_to_excel(weights): 
  # Get validation results
  results = calibrate.getValidationResults_multiobjective(weights)
  # Get the cases
  cases =[country+str(scenario) for country in case_studies for scenario in [1,2]]
  # Assign rows and columns names
  rows = validation_metrices
  cols = [gf+'_'+case for gf in mo_goal_functions for case in cases   ]
  # Save as excel file
  saveArrayAsExcel(results, 'validation_multiobjective_results_excel', sheet_name = 'val_results',
                   row_names=rows, col_names=cols)

def normalityTest(single_gf_metric, no_top_solutions, weights, alpha=0.05):
  """
  test of normality of distribuion of difference between the validation metrics using Shapiro-Wilk Test
  """
  # Get results for Kappa goal function
  results = reshapeSingleObjectiveResults_cases_scenarios_aggregated(single_gf_metric,no_top_solutions)
  # Get results for multiobjective goal functons
  mResults = reshapeMultiObjectiveResults_cases_scenarios_aggregated(no_top_solutions, weights)
  #create array to store test results
  a = np.empty((len(metricNames),len(validation_metrices)))
  # Loop multi objective goal functions:
  for mo,mo_goal in enumerate(mo_goal_functions):
    # Loop validation metrices:
    for v_m in range(len(validation_metrices)):
      # Create array containing differences
      d = results[:,v_m]-mResults[mo,:,v_m]
      # normality test
      stat, p = stats.shapiro(d)
      # fill the array
      if p > alpha:
        a[mo,v_m] = True
      else:
        a[mo,v_m] = False

def getDataframeStats(df, no_top_solutions):
  # Calculate stats:
  a_min = df.groupby(np.arange(len(df))//no_top_solutions).min()
  a_min.index=[ c+str(s)+'_min' for c in case_studies for s in [1,2] ]
  a_max = df.groupby(np.arange(len(df))//no_top_solutions).max()
  a_max.index=[ c+str(s)+'_max' for c in case_studies for s in [1,2] ]
  a_mean = df.groupby(np.arange(len(df))//no_top_solutions).mean()
  a_mean.index=[ c+str(s)+'_mean' for c in case_studies for s in [1,2] ]
  a_median = df.groupby(np.arange(len(df))//no_top_solutions).median()
  a_median.index=[ c+str(s)+'_median' for c in case_studies for s in [1,2] ]
  a_std = df.groupby(np.arange(len(df))//no_top_solutions).std()
  a_std.index=[ c+str(s)+'_std' for c in case_studies for s in [1,2] ]
  # Create df holding stats
  a_stats = pd.concat([a_min,a_max,a_mean,a_median,a_std])
  return a_stats
  
def saveTopSolutionsReshaped(single_gf_metric, no_top_solutions, weights):
  # Save single objective results
  results = reshapeSingleObjectiveResults_cases_scenarios_aggregated(single_gf_metric,no_top_solutions)
  # Assign name for single-objetive goal function
  table_name = 'h1(K)_'+str(no_top_solutions)+'_top_solutions_all_cases'
  # Assign row names
  rows = [ c+str(s)+'_'+str(i) for c in case_studies for s in [1,2] for i in range(no_top_solutions) ]
  # Convert the array to pd and save as excel file.
  saveArrayAsExcel(results, table_name, sheet_name = 'h1(K)', row_names=rows, col_names = validation_metrices)
  # Convert array to pd
  df = pd.DataFrame(results, index = rows, columns = validation_metrices)
  # Get descriptive stats
  stats = getDataframeStats(df, no_top_solutions)
  # Update the file
  appendArrayAsExcel(stats, table_name, sheet_name = 'descr_stats')
    
  # Save multi-objective results
  mResults = reshapeMultiObjectiveResults_cases_scenarios_aggregated(no_top_solutions, weights)
  for g,gf in enumerate(mo_goal_functions):
    # Convert the array to pd and save as excel file.
    table_name_mo = mo_goal_functions[g]+'_'+str(no_top_solutions)+'_top_solutions_all_cases'
    saveArrayAsExcel(mResults[g], table_name_mo,sheet_name = gf, row_names=rows, col_names = validation_metrices)
    # Convert array to pd
    df_mo = pd.DataFrame(results, index = rows, columns = validation_metrices)
    # Get descriptive stats
    stats_mo = getDataframeStats(df_mo, no_top_solutions)
    # Update the file
    appendArrayAsExcel(stats_mo, table_name_mo, sheet_name = 'descr_stats')

def statisticalTest(results, mResults, test):
  # Prepare empty array to store results: multiobjective functions x validation metrics x test results
  w_test = np.empty((len(metricNames),len(validation_metrices),2))
  # Loop multi objective goal functions:
  for mo,mo_goal in enumerate(mo_goal_functions):
    # Loop validation metrices:
    for v_m in range(len(validation_metrices)):
      w_test[mo,v_m] = sTest(results[:,v_m],mResults[mo,:,v_m],test)
  return w_test

def sTest(array0, array1, test):
  """
  returns: statistic, p-value
  """
  if test == 'Wilcoxon':
    s = np.array(stats.wilcoxon(array0,array1))
  elif test == 'MannWhitneyU':
    s = np.array(stats.mannwhitneyu(array0,array1)) #, alternative = 'greater'
  elif test == 'TTest_ind':
    s = np.array(stats.ttest_ind(array0,array1))
  elif test == 'Ansari_Bradley':
    s = np.array(stats.ansari(array0,array1))
  elif test == 'Kolmogorov_Smirnov_2s': #Kolmogorov-Smirnov statistic on 2 samples
    s = np.array(stats.ks_2samp(array0,array1))
  else:
    print('Choose another test')
    s= None
  return s

def getAbsoluteDifferenceArrays(gf_metric,val_metric,country,scenario,weights):
  """
  Get the differences between and modelled metrices for single- and multiobjective goal function
  Metric fdi and wfdi are measured for 16 zones, rest are measured for the whole case study area
  """ 
  # Get calibrated index of goal function based on Kappa
  singleIndex = calibrate.getCalibratedIndeks('kappa', scenario,case=country)
  # Get calibrated index of multiobjective goal function
  multiIndex = calibrate.getMultiObjectiveIndex(gf_metric, weights[0],weights[1],scenario, case=country)
  # Get the differences between observed and modelled metrices
  diffArray = calibrate.createDiffArray(val_metric, scenario, 'validation', case=country)
  # Get differences for calibrated Kappa goal function    
  singleDiffArray = diffArray[:,singleIndex]
  # Get differences for calibrated multiobjective goal function
  multiDiffArray = diffArray[:,multiIndex]
  # Get average values (for validation period) of differences for calibrated Kappa goal function
  avSingleDiffArray = calibrate.getAveragedArray(singleDiffArray, scenario, 'validation')
  # Get average values (for validation period) of differences for calibrated multiobjective goal function
  avMultiDiffArray = calibrate.getAveragedArray(multiDiffArray, scenario, 'validation')
  # Get absolute values of differences for results calibrated using Kappa goal function
  absSingleDiffArray = np.absolute(avSingleDiffArray)
  # Get absolute values of differences for results calibrated using mutlibjective goal function
  absMultiDiffArray = np.absolute(avMultiDiffArray)
  
  return absSingleDiffArray,absMultiDiffArray

def getLocationalMetricArrays(gf_metric,loc_metric, country,scenario,weights):
  """
  Get locational metric ('K','Ks','A') values for single- and multiobjective goal function
  """ 
  # Get calibrated index of goal function based on Kappa
  singleIndex = calibrate.getCalibratedIndeks('kappa', scenario,case=country)
  # Get calibrated index of multiobjective goal function
  multiIndex = calibrate.getMultiObjectiveIndex(gf_metric, weights[0],weights[1],scenario, case=country)
  # Get locational metric values
  if loc_metric == 'K':
    locationalArray = calibrate.getKappaArray(case=country)
  elif loc_metric == 'Ks':
    locationalArray = calibrate.getKappaSimulationArray(case=country)
  elif loc_metric == 'A':
    locationalArray = calibrate.getAllocationArray(case=country)
  # Get locational array fr single objective goal function parameter:
  singleLocationalArray = locationalArray[:,singleIndex]
  # Get locational array fr multiobjective goal function parameter:
  multiLocationalArray = locationalArray[:,multiIndex]
  # Get average values (for validation period) of singleobjective locational metric
  avSingleLocationalMetricArray = calibrate.getAveragedArray(singleLocationalArray, scenario, 'validation')
  # Get average values (for validation period) of multi objective locational metric
  avMultiLocationalMetricArray = calibrate.getAveragedArray(multiLocationalArray, scenario, 'validation')
  # Adjust the size to landscape metric and return
  return np.array([[avSingleLocationalMetricArray]]), np.array([[avMultiLocationalMetricArray]])

def getResultArray_all_cases(gf_metric, v_metric, weights):
  # Get case studies
  combinations = [[country,scenario] for country in case_studies for scenario in [1,2]]  
  alist0 = []
  alist1=[]
  # Fill the array
  for country, scenario in combinations:
    # Get the array depending on the validation metric type
    if v_metric in calibration_metrices[:-1]:
      a_tuple = getAbsoluteDifferenceArrays(gf_metric,v_metric,country,scenario,weights)
    else:
      a_tuple = getLocationalMetricArrays(gf_metric,v_metric,country,scenario,weights)
    # Improve this later
    alist0.append(a_tuple[0])
    alist1.append(a_tuple[1])
  array0 = np.array(alist0).flatten()
  array0 = array0[~np.isnan(array0)]
  array1 = np.array(alist1).flatten()
  array1 = array1[~np.isnan(array1)]
  # Create array to combine results for all cases
  outArray = np.empty((len(array0),2))
  outArray[:,0] = array0
  outArray[:,1] = array1
  return outArray

def errorStatistics(weights, alpha = 0.05):
  """
  Trial function
  Statistical test results of difference of metric errors between single and multiobjecctive goal functions
  All cases together
  NO SIGNIFICANT RESULTS
  """
  weights=[0.5,0.5]
  
  # Create array to store the results:
  statArray = np.empty((len(metricNames),len(validation_metrices)))
  for gf_m, v_m in [[gf_m, v_m] for gf_m in metricNames for v_m in validation_metrices ]:
    # Get results array (2D) with all cases combined. First column: single objective, second column: multiobjective:
    rArray = getResultArray_all_cases(gf_m,v_m,weights)
    # Create array containing differences
    d = rArray[:,0]-rArray[:,1]
    # normality test
    stat, p = stats.shapiro(d)
    # fill the array
    if p > alpha:
      # Normal distribution, TTest_ind
      test = 'TTest_ind'
    else:
      # Not normal distribution, MannWhitneyU test
      test = 'MannWhitneyU'

    # Run statistics
    sTestResults = sTest(rArray[:,0], rArray[:,1], test)
    
    if sTestResults[1] < alpha:
      print('gf: K+',gf_m,'validated on:',v_m)
      print(sTestResults)
      print('significant')
      print(np.mean(rArray[:,0]))
      print(np.mean(rArray[:,1]))

def zonalErrorStatistics(weights, test, alpha = 0.05):
  """
  Trial function for statistical test based on errors, only zonal statistics
  Each case study separate
  """
  # Create the figure
  fig, axs = plt.subplots(6,2, figsize=(7.14,17))
  # Metrics calculated for 16 zones
  j=0
  for m,metric in enumerate(['fdi','wfdi']):
    i=0
    for country in case_studies:
      for scenario in [1,2]:
        rSingleResults = []
        rMultiResults = []
        
        singleIndex = calibrate.getCalibratedIndeks('kappa', scenario,case=country)
        multioIndex = calibrate.getMultiObjectiveIndex(metric, weights[0],weights[1],scenario, case=country)
        diffArray = calibrate.createDiffArray(metric, scenario, 'validation', case=country)
        singleDiffArray = diffArray[:,singleIndex]
        multioDiffArray = diffArray[:,multioIndex]
        avSingleDiffArray = calibrate.getAveragedArray(singleDiffArray, scenario, 'validation')
        avMultioDiffArray = calibrate.getAveragedArray(multioDiffArray, scenario, 'validation')
        rSingleResults.append(avSingleDiffArray)
        rMultiResults.append(avMultioDiffArray)
        
        # Reshape (combine all zones)
        rSingleResults = [ rSingleResults[i].flatten() for i in range(len(rSingleResults)) ]
        rSingleResults = np.reshape(np.array(rSingleResults),(1*16,1))
        rMultiResults = [ rMultiResults[i].flatten() for i in range(len(rMultiResults)) ]
        rMultiResults = np.reshape(np.array(rMultiResults),(1*16,1))
        # Some zones might be nan. Remove them.
        rSingleResults = rSingleResults[~np.isnan(rSingleResults)]
        rMultiResults = rMultiResults[~np.isnan(rMultiResults)]
        # Get the absolute value of difference between modelled and observed
        rSingleResults = np.absolute(rSingleResults)
        rMultiResults = np.absolute(rMultiResults)
       
        ##############################################################
        # Fixing the bug from scipy code:
        if test == 'MannWhitneyU':
          x = np.asarray(rSingleResults)
          y = np.asarray(rMultiResults)
          n1 = len(x)
          n2 = len(y)
          ranked = stats.rankdata(np.concatenate((x, y)))
          rankx = ranked[0:n1]  # get the x-ranks
          ranky = ranked[n1:-1]
          u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rankx, axis=0)  # calc U for x
          u2 = n1*n2 - u1  # calc U for x
          u = min(u1,u2)
        ##############################################################
        
        # Run statistics
        sTestResults = sTest(rSingleResults, rMultiResults, test)
        p_value = sTestResults[1]
        # Get statistics
        median_singlo = np.median(rSingleResults)
        median_multio = np.median(rMultiResults)
        # test the improvement of errors
        impr = np.sign(median_singlo - median_multio)
        # assign colors to improvement:
        impr_c = {1: 'mediumseagreen', -1: 'indianred', 0:'none', -0:'none'}
        # Get Ansari-Bradley test result for equal scale parameters
        equal_scale_test = sTest(rSingleResults, rMultiResults, 'Ansari_Bradley')
        # Create text with statistics as labels
        textstr = '\n'.join(
          ("h1(K):",
           "min: = {0:.2E}".format(np.amin(rSingleResults)),
           "max: = {0:.2E}".format(np.amax(rSingleResults)),
           "mean: = {0:.2E}".format(np.mean(rSingleResults)),
           "median: = {0:.2E}".format(np.median(rSingleResults)),
           "sd: = {0:.2E}".format(np.std(rSingleResults)),
           "var: = {0:.2E}".format(np.var(rSingleResults)),
           '\n'+'f('+metric.upper()+') and h1(K)'+':',
           "min: = {0:.2E}".format(np.amin(rMultiResults)),
           "max: = {0:.2E}".format(np.amax(rMultiResults)),
           "mean: = {0:.2E}".format(np.mean(rMultiResults)),
           "median: = {0:.2E}".format(np.median(rMultiResults)),
           "sd: = {0:.2E}".format(np.std(rMultiResults)),
           "var: = {0:.2E}".format(np.var(rMultiResults)),
           "\n"+"n: {}".format(len(rSingleResults)),
           #"Ansari-Bradley test",
           #"p_value: {0:.2E}".format(equal_scale_test[1]),
           test+' Test:',
           #"statistic: {0:.2E}".format(s_test[mo,v_m,0]), # does not apply for n>20
           "p_value: {0:.2E}".format(p_value)))
        # Create two boxplots:
        axs[i,j].boxplot(
          [rSingleResults,rMultiResults],positions=[1.5,2],patch_artist=True)
        # If significant, change the background:
        if p_value < alpha:
          axs[i,j].set_facecolor(impr_c[impr])
        # Assign ylabel
        axs[i,j].set_ylabel(metric_units[m])
        # Adjust ticks
        axs[i,j].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        # Place a textbox
        axs[i,j].text(0.02,0.02,textstr, transform=axs[i,j].transAxes, fontsize=6)
        # Extend the plot
        axs[i,j].set_xlim(left=0)
        # Assign ticks
        axs[i,j].set_xticklabels(['h1(K)','f('+metric.upper()+') and h1(K)'])

        i+=1
    j+=1
  # Add titles for validation metrics:
  for v_m, v_metric in enumerate([validation_metrices[i] for i in [1,2]]):
    axs[0,v_m].text(0.5,1.05,v_metric, transform=axs[0,v_m].transAxes,
                     fontsize=10, horizontalalignment='center', weight='bold')
  # Add titles for case studies:
  combinations = [country+str(scenario) for country in case_studies for scenario in [1,2]]
  for c,case_study in enumerate(combinations):
    axs[c,0].text(-0.25,0.5,case_study, transform=axs[c,0].transAxes,
                     fontsize=10, verticalalignment='center', weight='bold', rotation = 'vertical')

  setNameClearSave('stats_zonal_fdi_wfdi_best_solution_only_'+test+\
                   '_'+str(int(weights[0]*10))+'_'+str(int(weights[1]*10)))

      
#zonalErrorStatistics([0.5,0.5],'Kolmogorov_Smirnov_2s')


#############################
########### PLOTS ###########
#############################

def plotStatisticalTest(no_top_solutions, weights, test, kappa_gf  = True, alpha=0.05):
  """
  Generates two boxplots for a validation metric (all case studies aggregated)
  first for the goal function Kappa, second for the mutliobjective goal function
  Plots statistics and test results on the side

  test in ['MannWhitneyU','TTest_ind','Wilcoxon, 'Kolmogorov_Smirnov_2s'']
  weights = [w_RMSE, w_Kappa]
  """
  # Create the figure
  fig, axs = plt.subplots(8,11, figsize=(30,15))
  # Get results for multiobjective goal functons
  #old mResults = reshapeMultiObjectiveResults_cases_scenarios_aggregated(no_top_solutions, weights)
  mResults = reshapeMultiObjectiveResults_scenarios_aggregated(no_top_solutions, weights)
  # Ih the single objective goal function is based on Kappa, get the values here:
  if kappa_gf is True:
    single_gf_metric = 'K'
    #old results = reshapeSingleObjectiveResults_cases_scenarios_aggregated(single_gf_metric,no_top_solutions)
    # Get results 3D array: case studies x top performers x val metrics:
    results = reshapeSingleObjectiveResults_scenarios_aggregated(single_gf_metric,no_top_solutions)
  # Loop multi objective goal functions:
  for mo,mo_goal in enumerate(mo_goal_functions):
    # Get landscape metric used in the multi-objective goal function
    if kappa_gf != True:
      single_gf_metric = calibration_metrices[mo]
      # Get results for single-objective goal function based on the landscape metric
      #old results = reshapeSingleObjectiveResults_cases_scenarios_aggregated(single_gf_metric,no_top_solutions)
      results = reshapeSingleObjectiveResults_scenarios_aggregated(single_gf_metric,no_top_solutions)
      
    '''As we are using Kolmogorov test, no Ansari test is needed
    # Get Ansari-Bradley test result for equal scale parameters
    equal_scale_test = statisticalTest(results, mResults, 'Ansari_Bradley')
    '''
    # Loop validation metrices and case studies:
    for v_m, c in [(v_m,c) for v_m in range(len(validation_metrices)) for c in range(len(case_studies))]:
      # Get statistical test results for all goal functions and metrics
      s_test = statisticalTest(results[c], mResults[:,c], test)
      # check if the test was significant:
      p_value = s_test[mo,v_m,1]
      # get the median of the error for the validation metric c_m for all performers 
      median_singlo = np.median(results[c,:,v_m])
      median_multio = np.median(mResults[mo,c,:,v_m])
      # test the improvement of errors
      impr = np.sign(median_singlo - median_multio)
      # reverse for metrics showing accuracy:
      if v_m in [8,9]:
        impr = -impr
      #  Create text with statistics as labels
      textstr = '\n'.join(
        ("country: "+case_studies[c],
         goal_functions[mo],
         "min: = {0:.2E}".format(np.amin(results[c,:,v_m])),
         "max: = {0:.2E}".format(np.amax(results[c,:,v_m])),
         #"mean: = {0:.2E}".format(np.mean(results[c,:,v_m])),
         "median: = {0:.2E}".format(np.median(results[c,:,v_m])),
         "var: = {0:.2E}".format(np.var(results[c,:,v_m])),
         mo_goal+':',
         "min: = {0:.2E}".format(np.amin(mResults[mo,c,:,v_m])),
         "max: = {0:.2E}".format(np.amax(mResults[mo,c,:,v_m])),
         #"mean: = {0:.2E}".format(np.mean(mResults[mo,c,:,v_m])),
         "median: = {0:.2E}".format(np.median(mResults[mo,c,:,v_m])),
         "var: = {0:.2E}".format(np.var(mResults[mo,c,:,v_m])),
         test+' Test:',
         #"statistic: {0:.2E}".format(s_test[mo,v_m,0]), # does not apply for n>20
         "p_value: {0:.2E}".format(p_value)))
      # Create two boxplots:
      bp = axs[mo,v_m].boxplot(
        [results[c,:,v_m],mResults[mo,c,:,v_m]],positions=[1.5+c/2,3.5+c/2],patch_artist=True)
      # If significant, change the background and the boxplot alpha:
      if p_value < alpha:
        axs[mo,v_m].set(facecolor='thistle', alpha=0.5)
        plt.setp(axs[mo,v_m].spines.values(), color='rebeccapurple', lw=3)
        bp_alpha = 0.8
      else:
        bp_alpha = 0.25
      # Assign ylabel
      axs[mo,v_m].set_ylabel(metric_units[v_m])
      # Adjust ticks
      axs[mo,v_m].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
      # Place a textbox
      #axs[mo,v_m].text(0.02,0.02+c*0.35,textstr, transform=axs[mo,v_m].transAxes, fontsize=3)
      # Place text woth number of observations
      axs[mo,v_m].text(0.9,0.96,"n: {}".format(len(results[c])),transform=axs[mo,v_m].transAxes, fontsize=6)
      # Extend the plot
      axs[mo,v_m].set_xlim(left=.75)
      # Get the tick for single-objective goal function
      if kappa_gf is True:
        s_tick = 'h1(K)'
      else:
        s_tick = goal_functions[mo]
      # Assign ticks
      axs[mo,v_m].set_xticks([2,4])
      axs[mo,v_m].set_xticklabels([s_tick,mo_goal])
      # Assign colors and transparency to boxplots, fliers and medians
      for box in bp['boxes']:
        box.set_facecolor(countryColors[case_studies[c]])
        box.set_alpha(bp_alpha)
      for flier in bp['fliers']:
        flier.set(marker='o', color='black', markersize=2)
        flier.set_alpha(bp_alpha)
      for median in bp['medians']:
        median.set(linestyle='dashed',color='#505050')
        median.set_alpha(bp_alpha)
      for whisker in bp['whiskers']:
        whisker.set_alpha(bp_alpha)
      for cap in bp['caps']:
        cap.set_alpha(bp_alpha)
          
  # Add titles for validation metrics:
  for v_m, v_metric in enumerate(validation_metrices):
    axs[0,v_m].text(0.5,1.05,v_metric, transform=axs[0,v_m].transAxes,
                     fontsize=10, horizontalalignment='center', weight='bold')
  # Add titles for multi objective goal functions:
  for mo, mo_f in enumerate(mo_goal_functions):
    axs[mo,0].text(-0.2,0.5,mo_f, transform=axs[mo,0].transAxes,
                     fontsize=10, verticalalignment='center', weight='bold', rotation = 'vertical')
      
  K_end = {True:'K',False:'landscape_metric'}
  setNameClearSave('stats_'+test+str(no_top_solutions)+'_single-objective_'+K_end[kappa_gf])
#plotStatisticalTest(17,[0.6,0.4],'Kolmogorov_Smirnov_2s')

def plotUrbObs(): # Done
  """
  Plot the observed urban areas in every observation year
  15 subplots stacked in 3x5 matrix
  """
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(3,5, sharex=True, sharey=True, figsize=(7.14,4)) # 16 cm of width
  cmap = colors.ListedColormap(['black','yellow'])
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  ## 2. Loop countries to get the data
  for c,country in enumerate(case_studies):
    urb_obs = calibrate.getObservedArray('urb', case=country)
    # Plot the observed urban areas
    for index, oYear in enumerate(observedYears):
      axs[c,index].get_xaxis().set_ticks([])
      axs[c,index].get_yaxis().set_ticks([])
      axs[c,index].xticks = ([])
      # Reshape the 1D array into a map
      urbMatrix1 = np.reshape(urb_obs[index,0,1], (1600,1600))
      axs[c,index].imshow(urbMatrix1, cmap=cmap)
      axs[c,index].set(xlabel=oYear, ylabel=country)
      axs[c,index].label_outer()
  
  setNameClearSave('urb_observed')

def plotUrbChanges(): # add legend
  """
  Plot the observed urban changes in every pair of observation year
  12 subplots stacked in 3x4 matrix
  """
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(3,4, sharex=True, sharey=True, figsize=(5.7,4)) # size to match plotUrbObs()
  cmap = colors.ListedColormap(c_obs)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  ## 2. Loop countries to get the data
  for c,country in enumerate(case_studies):
    urb_obs = calibrate.getObservedArray('urb', case=country)
    
    # Plot the observed urban areas
    for index, oYear in enumerate(observedYears[:-1]):
      axs[c,index].get_xaxis().set_ticks([])
      axs[c,index].get_yaxis().set_ticks([])
      axs[c,index].xticks = ([])
      # Reshape the 1D array into a map
      changeMatrix = urb_obs[index+1,0,1] - urb_obs[index,0,1] 
      changeMatrix_reshape = np.reshape(changeMatrix, (1600,1600))
      axs[c,index].imshow(changeMatrix_reshape, cmap=cmap)
      axs[c,index].set(xlabel=str(oYear) + '-' + str(observedYears[index+1]), ylabel=country)
      axs[c,index].label_outer()
  
  setNameClearSave('urb_changes_observed')

def plotUrbMod(): #add legend
  """
  Plot the modelled urban areas in observation year 2018 (last observation year)
  for all metrics, 3 case studies and 2 scenarios
  30 subplots stacked in 6x5 matrix
  """
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(6,5, sharex=True, sharey=True, figsize=(7.14,9)) # 16 cm of width
  cmap = colors.ListedColormap([c_mod[1],c_mod[2]])
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  ## 2. Loop and plot countries
  i=0
  for c,country in enumerate(case_studies):
    for scenario in [1,2]:
      j=0
      for metric in all_metrices:
        theIndex = calibrate.getCalibratedIndeks(metric,scenario,country)
        # Get the data for the last observation year
        urb_mod = calibrate.getModelledArray('urb_subset_'+str(obsTimeSteps[-1]),case=country)
        # Reshape the 1D array into a map
        urbMatrix1 = np.reshape(urb_mod[0,len(observedYears),1], (1600,1600))
        # Plot
        axs[i,j].imshow(urbMatrix1, cmap=cmap)
        if metric == 'kappa':
          axs[i,j].set(xlabel=metric[0].upper(), ylabel=country+str(scenario))
        else:
          axs[i,j].set(xlabel=metric.upper(), ylabel=country+str(scenario))            
        j+=1
      i+=1
        
  # Adjusr ticks
  for ax in axs.flat:
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.xticks = ([])
    ax.label_outer()
      
  setNameClearSave('urb_modelled')

def plotUrbChangesMod(scenario): #DONE
  """
  Plots observed and modelled changes between years 2000 and 2006
  First column on the left presents the observed chages.
  18 subplots stacked in 3x6 matrix.
  """
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(3,10, sharex=True, sharey=True, figsize=(7.6,4)) # 16 cm of width
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  # Set colors
  colorsModelledChange = {-1: c_mod[0],0: c_mod[1],1: c_mod[2]}
  cmap_obs = colors.ListedColormap(c_obs)
  ## 2. Loop and plot modellled countries
  # Select the time steps
  period = parameters.getCalibrationPeriod()[scenario]['validation'] # scenario 1 calibration: [1,2]
  selected_time_steps = np.array(parameters.getObsTimesteps())[period] #[1,11,17,23,29]

  i=0
  for country in case_studies:
    # And add the map for the observed change! Get the data:
    urb_obs = calibrate.getObservedArray('urb', case=country)
    changeMatrix = urb_obs[period[1],0,1] - urb_obs[period[0],0,1]
    # Reshape the 1D array into a map
    changeMatrix_reshape = np.reshape(changeMatrix, (1600,1600))
    # Plot the observed changes
    axs[i,0].imshow(changeMatrix_reshape, cmap=cmap_obs)
    axs[i,0].set(xlabel='observed', ylabel=country+str(scenario))
    # Loop metrics and for each metric get the change map for seected country, and selected metric (goal function)    
    j=1
    for m, metric in enumerate(all_metrices):
      calibratedIndex = calibrate.getCalibratedIndeks(metric,scenario,case=country)
      # Get the data for period:
      # first year between observation yuears to last year in observation years
      '''urb_mod_current = calibrate.getModelledArray(
        'urb_subset_'+str(selected_time_steps[0]),case=country)
      print(urb_mod_current.shape)'''
      urb_mod_next = calibrate.getModelledArray(
        'urb_subset_'+str(selected_time_steps[1]),case=country)          
      changeModMatrix = (urb_mod_next[0,calibratedIndex,1]
                      - urb_obs[period[0],0,1])
      # Reshape the matrix into 2D
      changeModMatrix_reshape = np.reshape(changeModMatrix, (1600,1600))
      # Find states in the map to adjust the colors:
      u = np.unique(changeModMatrix[~np.isnan(changeModMatrix)])
      cmapL = [colorsModelledChange[v] for v in u]
      cmap = colors.ListedColormap(cmapL)
      # Plot
      axs[i,j].imshow(changeModMatrix_reshape, cmap=cmap)
      axs[i,j].set(xlabel=goal_functions[m], ylabel=country+str(scenario))
      
      j+=1
    i+=1

  # Adjusr ticks
  for ax in axs.flat:
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.xticks = ([])
    ax.label_outer()

  # Create two legends. One for observed, one for modelled
  labels = {0:['Observed:','urban to non-urban','no change','non-urban to urban'],
            1:['Modelled:','urban to non-urban','no change','non-urban to urban']}
  for i,c in enumerate([c_obs,c_mod]):
    # add blank color to the color list
    c = ['none']+c
    # Make patches:
    patches = [ mpatches.Patch(color=c[j], label="{l}".format(l=labels[i][j]) ) for j in range(len(labels[i])) ]
    # Set a legend's anchor
    anchor = {0:0,1:3}
    leg = axs[0,anchor[i]].legend(
      handles=patches,
      bbox_to_anchor=(0., 1.1, 3+2*0.1, .102),
      loc='lower center',
      ncol=2,
      mode="expand",
      borderaxespad=0.,
      fontsize=6)
    leg.get_frame().set_edgecolor('darkviolet')
    leg.get_frame().set_linewidth(0.50)

  setNameClearSave('urb_changes_modelled_'+str(scenario))
  
def plotTopSolutions(no_top_solutions): #Done
  """
  Plot 9 subplots, each for a goal function.
  Each subplot shows boxplots for best parameter sets, for 3 case studies and 2 scenarios
  """
  nrOfBestPerformers = no_top_solutions #~15% of all parameters
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(9, figsize=(8,10),sharex=True) # 16 cm of width 
  plt.subplots_adjust(wspace=0.1, hspace=0.25)
  # Add y label
  fig.text(0.06, 0.5, "parameters for "+str(nrOfBestPerformers)+' best solutions', va='center', rotation='vertical')
  # Add x label
  plt.xlabel('drivers')
  
  # set the width and the space
  width = 0.13 # width of a bar
  space = 0.02 # space between case studies
  ind = np.arange(4)+1 # four drivers
  alpha={1:0.6,2:0.3}
  # Set labels
  drivers = ['NEIGH', 'TRAIN', 'TRAVEL', 'LU']
  # Create a list of boxplots
  boxplots = []
  i=0
  
  for m_i,metric in enumerate(all_metrices): #metricNames for multiobjective
    j=0
    for c_i,country in enumerate(case_studies):
      # set the color for the plot, dependent on the country:
      c = countryColors[country]
      for scenario in [1,2]:
        # Assign position:
        positionDict = {
          0: ind - (2.5*width+space),
          1: ind - (1.5*width+space),
          2: ind - 0.5*width,
          3: ind + 0.5*width,
          4: ind + (1.5*width+space),
          5: ind + (2.5*width+space)}
        # Change the colors transparency
        c_new = colors.to_rgba(c)[:-1]+(alpha[scenario],)
        # Get the top performing parameters
        topperformers = calibrate.getTopCalibratedParameters(metric,scenario,nrOfBestPerformers,country)
        # multiobjective:
        #topperformers = calibrate.getTopCalibratedParameters_multiobjective(metric,[0.5,0.5],scenario,no_top_solutions,country)
        y = topperformers[:,3:].astype('float64')
        print(metric,scenario,country)
        np.set_printoptions(precision=2, linewidth=100) 
        print(np.median(y,axis=0))

        # Set subplot title
        axs[i].set_title(goal_functions[m_i], pad=2)
        # Plot boxplot
        bp = axs[i].boxplot(
          y,
          whis=[5,95],# set the whiskers at specific percentiles of the data
          widths=width,
          positions=positionDict[j],
          patch_artist=True)
        # Assign colors and transparency to boxplots, fliers and medians
        for box in bp['boxes']:
          box.set_facecolor(c_new)
          #box.set_alpha(0.5)
        for flier in bp['fliers']:
          flier.set(marker='o', color=c_new, alpha=0.5, markersize=2)
        for median in bp['medians']:
          median.set(linestyle='dashed',color='#505050')
          #median.set_color('darkgrey')
        j+=1
        if m_i==0:
          boxplots.append(bp)
    i+=1
  # Set the ticks and ticklabels for all axes
  plt.setp(axs, xticks=ind, xticklabels=drivers)#,yticks=[1, 2, 3])
  # Set the limits
  for ax in axs:
    ax.set_xlim(0.4, 4.6)
  # Create a legend:
  leg = ax.legend([ box['boxes'][0] for box in boxplots ],
            [x+str(y) for x in case_studies for y in [1,2]],
            bbox_to_anchor=(0., 1.2, 1, .102),
            loc='lower center',
            ncol=6,
            mode="expand",
            bbox_transform=axs[0].transAxes,
            borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  
  setNameClearSave(str(nrOfBestPerformers)+'_top_solutions')
#print('here')

def plotMultiobjectiveTopSolutions(no_top_solutions): #Done
  """
  Plot 8 subplots, each for a multiobjective goal function.
  Each subplot shows boxplots for best parameter sets, for 3 case studies and 2 scenarios
  """
  nrOfBestPerformers = no_top_solutions #~15% of all parameters
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(8, figsize=(8,8),sharex=True) # 16 cm of width 
  plt.subplots_adjust(wspace=0.1, hspace=0.25)
  # Add y label
  fig.text(0.06, 0.5, "parameters for "+str(nrOfBestPerformers)+' top multiobjective solutions', va='center', rotation='vertical')
  # Add x label
  plt.xlabel('drivers')
  
  # set the width and the space
  width = 0.13 # width of a bar
  space = 0.02 # space between case studies
  ind = np.arange(4)+1 # four drivers
  alpha={1:0.6,2:0.3}
  # Set labels
  drivers = ['NEIGH', 'TRAIN', 'TRAVEL', 'LU']
  # Create a list of boxplots
  boxplots = []
  i=0
  
  for m_i,metric in enumerate(metricNames):
    j=0
    for c_i,country in enumerate(case_studies):
      # set the color for the plot, dependent on the country:
      c = countryColors[country]
      for scenario in [1,2]:
        # Assign position:
        positionDict = {
          0: ind - (2.5*width+space),
          1: ind - (1.5*width+space),
          2: ind - 0.5*width,
          3: ind + 0.5*width,
          4: ind + (1.5*width+space),
          5: ind + (2.5*width+space)}
        # Change the colors transparency
        c_new = colors.to_rgba(c)[:-1]+(alpha[scenario],)
        # Get the top performing parameters
        #topperformers = calibrate.getTopCalibratedParameters(metric,scenario,nrOfBestPerformers,country)
        # multiobjective:
        topperformers = calibrate.getTopCalibratedParameters_multiobjective(metric,[0.5,0.5],scenario,no_top_solutions,country)
        #y = topperformers[:,3:].astype('float64')
        y = topperformers[:,4:].astype('float64') # multiobjective

        # Set subplot title
        axs[i].set_title(mo_goal_functions[m_i], pad=2)
        # Plot boxplot
        bp = axs[i].boxplot(
          y,
          whis=[5,95],# set the whiskers at specific percentiles of the data
          widths=width,
          positions=positionDict[j],
          patch_artist=True)
        # Assign colors and transparency to boxplots, fliers and medians
        for box in bp['boxes']:
          box.set_facecolor(c_new)
          #box.set_alpha(0.5)
        for flier in bp['fliers']:
          flier.set(marker='o', color=c_new, alpha=0.5, markersize=2)
        for median in bp['medians']:
          median.set(linestyle='dashed',color='#505050')
          #median.set_color('darkgrey')
        j+=1
        if m_i==0:
          boxplots.append(bp)
    i+=1
  # Set the ticks and ticklabels for all axes
  plt.setp(axs, xticks=ind, xticklabels=drivers)#,yticks=[1, 2, 3])
  # Set the limits
  for ax in axs:
    ax.set_xlim(0.4, 4.6)
  # Create a legend:
  leg = ax.legend([ box['boxes'][0] for box in boxplots ],
            [x+str(y) for x in case_studies for y in [1,2]],
            bbox_to_anchor=(0., 1.2, 1, .102),
            loc='lower center',
            ncol=6,
            mode="expand",
            bbox_transform=axs[0].transAxes,
            borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  
  setNameClearSave(str(nrOfBestPerformers)+'_multiobjective_top_solutions')

def plotTopSolutionsStats(no_top_solutions): #Done
  """
  Plot seven subplots, each presentnig a validation metric value obtained using a Kapa goal function
  Each subplot shows boxplots with errors for best parameter sets, for 3 case studies and 2 scenarios
  """
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(8,2, figsize=(8,8),sharex=True) # 16 cm of width
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  # Add y label
  fig.text(0.02, 0.5, "validation results for  "+str(no_top_solutions)+' best solutions for goal function h1(K)', va='center', rotation='vertical')
  # set the width and the space
  width = 0.2 # width of a bar
  space = 0.05 # space between case studies
  # set ticks position
  ind = np.array([1]) # only one goal function per plot
  # Assign position:
  positions = [1 - (2.5*width+space),
               1 - (1.5*width+space),
               1 - 0.5*width,
               1 + 0.5*width,
               1 + (1.5*width+space),
               1 + (2.5*width+space)]
  # Get colors      
  colors = getCountryColors()
  # Create a list of boxplots
  boxplots = []
  # Get the results for the top performing parameters
  results = getSingleGoalFunctionStats('K',no_top_solutions)
  # Loop validation metrics to plot each subplot
  i=0
  k=0
  for m_i,metric in enumerate(validation_functions):
    # Control the subplot positions
    if k>1:
      k=0
      i+=1
    # Set subplot title
    axs[i,k].set_title(validation_metrices[m_i], pad=2)
    # Set subplot y label
    axs[i,k].set_ylabel(metric_units[m_i])        
    # Get the value of the validation error for a given country and case study
    y = [r for r in results[:,:,m_i]]
    # Plot boxplot
    bp = axs[i,k].boxplot(
      y,
      whis=[5,95],# set the whiskers at specific percentiles of the data
      widths=width,
      positions=positions,
      patch_artist=True)
    # Set x limits:
    axs[i,k].set_xlim(0.1, 1.9)
    # Assign colors boxplots
    b=0
    for box in bp['boxes']:
      if b>5:
        b=0
      box.set_facecolor(colors[b])
      b+=1
    # Assign transparency to fliers
    for flier in bp['fliers']:
      flier.set(marker='o', alpha=0.5, markersize=2)
    # Assign colors and transparency to medians
    for median in bp['medians']:
      median.set(linestyle='dashed',color='#505050')
    # Remove ticks
    axs[i,k].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off
    if m_i==0:
      boxplots.append(bp)
    k+=1
  # Delete last subplot:
  fig.delaxes(axs[3,1])
  # Make patches for the legend:
  labels = [x+str(y) for x in case_studies for y in [1,2]]
  patches = [ mpatches.Patch(color=colors[j], label="{l}".format(l=labels[j]) ) for j in range(len(labels)) ]
  leg = axs[0,0].legend(
    handles=patches,
    labels=labels,
    bbox_to_anchor=(0., 1.2, 2+0.2, .102),
    loc='lower center',
    ncol=6,
    mode="expand",
    bbox_transform=axs[0,0].transAxes,
    borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  
  setNameClearSave('validation_results_for_h1(K)_'+str(no_top_solutions)+'_top_solutions_stats')

#plotTopSolutionsStats(17) # doesn't look nice now

def plotMultiobjectiveTopSolutionsStats(no_top_solutions, weights): #Done
  """
  Produce and save 4 plots, each for a goal function used with Kappa in calibration
  Plot seven subplots, each presentnig a validation metric value obtained using a given goal function
  Each subplot shows boxplots with errors for 15% of best parameter sets, for 3 case studies and 2 scenarios

  weights = [w_RMSE, w_Kappa]
  """
  # Get the results for the top performing parameters
  multio_results = getMultiobjectiveGoalFunctionStats(no_top_solutions, weights)
  # Loop the goal function landscape metrics, to get each plot
  for gf, gf_metric in enumerate(metricNames):
    ## 1. Prepare the plot  
    fig, axs = plt.subplots(4,2, figsize=(8,8),sharex=True) # 16 cm of width
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # Add y label
    fig.text(0.02, 0.5,
             "validation results for  "+str(no_top_solutions)+\
             ' best solutions for multiobjective goal function '+goal_functions[gf]+' and h1(K)',
             va='center', rotation='vertical')
    # set the width and the space
    width = 0.2 # width of a bar
    space = 0.05 # space between case studies
    # set ticks position
    ind = np.array([1]) # only one goal function per plot
    # Assign position:
    positions = [1 - (2.5*width+space),
                 1 - (1.5*width+space),
                 1 - 0.5*width,
                 1 + 0.5*width,
                 1 + (1.5*width+space),
                 1 + (2.5*width+space)]
    # Get colors      
    colors = getCountryColors()
    # Create a list of boxplots
    boxplots = []
    # Get the results for the top performing parameters
    results = multio_results[gf]
    # Loop validation metrics to plot each subplot
    i=0
    k=0
    for m_i,metric in enumerate(validation_functions):
      # Control the subplot positions
      if k>1:
        k=0
        i+=1
      # Set subplot title
      axs[i,k].set_title(validation_metrices[m_i], pad=2)
      # Set subplot y label
      axs[i,k].set_ylabel(metric_units[m_i])
      # Adjust ticks
      axs[i,k].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
      # Get the value of the validation error for a given country and case study
      y = [r for r in results[:,:,m_i]]
      # Plot boxplot
      bp = axs[i,k].boxplot(
        y,
        whis=[5,95],# set the whiskers at specific percentiles of the data
        widths=width,
        positions=positions,
        patch_artist=True)
      # Set x limits:
      axs[i,k].set_xlim(0.1, 1.9)
      # Assign colors boxplots
      b=0
      for box in bp['boxes']:
        if b>5:
          b=0
        box.set_facecolor(colors[b])
        b+=1
      # Assign transparency to fliers
      for flier in bp['fliers']:
        flier.set(marker='o', alpha=0.5, markersize=2)
      # Assign colors and transparency to medians
      for median in bp['medians']:
        median.set(linestyle='dashed',color='#505050')
      # Remove ticks
      axs[i,k].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
      if m_i==0:
        boxplots.append(bp)
      k+=1
    # Delete last subplot:
    fig.delaxes(axs[3,1])
    # Make patches for the legend:
    labels = [x+str(y) for x in case_studies for y in [1,2]]
    patches = [ mpatches.Patch(color=colors[j], label="{l}".format(l=labels[j]) ) for j in range(len(labels)) ]
    leg = axs[0,0].legend(
      handles=patches,
      labels=labels,
      bbox_to_anchor=(0., 1.2, 2+0.2, .102),
      loc='lower center',
      ncol=6,
      mode="expand",
      bbox_transform=axs[0,0].transAxes,
      borderaxespad=0.)
    leg.get_frame().set_edgecolor('darkviolet')
    leg.get_frame().set_linewidth(0.50)
    
    setNameClearSave('validation_results_for_h1(K)_and_'+goal_functions[gf]+\
                     '_'+str(no_top_solutions)+'_top_solutions_stats')

def plotGoalFunctionEverySet(): #DONE
  """
  Plot the values for a goal function durnig calibration. Values for every parameter set tested
  9 subplots stacked vertically, one for every metric
  """

  # First, get all results
  results = getAverageResultsArrayEverySet('calibration')
  # Loop the data for all metrics to get minimum and maximum goal function values
  limits = {}
  for i,m in enumerate(calibration_metrices):
    limits[m] = {
      'min': np.amin(results[i]),
      'max': np.amax(results[i]),
      'mean': np.mean(results[i]),
      'median':np.median(results[i]),
      'sd':np.std(results[i])
      }

  # Now, plot the data  
  ## 1. Get the parameter sets
  parameterSets = calibrate.getParameterConfigurations()
  parameters=np.arange(0,len(parameterSets),1)
  n = len(metricNames)+1 # number of subplots
  
  ## 2. Prepare the plot  
  fig, axs = plt.subplots(n, figsize=(7.14,7), sharex = True) # 16 cm of width
  xticks = np.arange(0, parameters[-1]+10, 15.0)
  plt.xticks(xticks,[int(x) for x in xticks])
  plt.xlabel('parameter set')
  fig.align_ylabels()
  plt.subplots_adjust(hspace=0.4)
  
  ## 3. Loop metrics. Each metric = new subplot
  for i,m in enumerate(calibration_metrices):
    j=0
    axs[i].set_ylabel(metric_units[i])  
    # Loop all the countries. Each suplot has data for all case studies:
    for country in case_studies:
      # Loop calibration scenarios:
      for scenario in [1,2]:
        # set the min and max y axis values:
        amin = limits[m]['min']
        amax = limits[m]['max']
        axs[i].set_title(goal_functions[i], pad=2)
        axs[i].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        axs[i].set_ylim([amin,amax])
        axs[i].set_yticks([amin,amax])
        # Create the labels only for one metric
        if i>0:
          myLabel = {1:None,2:None}
        else:
          myLabel = {1:country+str(scenario),2:country+str(scenario)}
        fmt = {1:'-',2:'--'}
        # plot
        axs[i].plot(
          parameters,
          results[i,j],
          fmt[scenario],
          linewidth = 0.5,
          label = myLabel[scenario],
          c = countryColors[country])
        # Plot a line showing mean metric vaue in the parameter space and the value
        axs[i].axhline(y=limits[m]['mean'], alpha=0.2,c='black',linestyle='--', linewidth=0.8)
        axs[i].text(300,limits[m]['mean']*0.9,
                    'mean = '+str(np.format_float_scientific(limits[m]['mean'],precision=2)),
                    fontsize=6)
        j+=1

  # Create the legend
  leg = fig.legend(
    bbox_to_anchor=(0., 1.35, 1, .102),
    loc='lower center',
    ncol=6,
    mode="expand",
    bbox_transform=axs[0].transAxes,
    borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  
  # Set the name and clear the directory if needed
  setNameClearSave('calibration_all_sets', scenario=None)

def plotParameters(): #DONE
  """
  Plot bars presenting parameters (0-1, y axis) for 5 goal funcstions (x axis)
  There are 5 metrics, 3 case studies, 2 scenarios --> 15 bars
  Ignore the warning
  """
 
  # Create figure
  fig = plt.figure(figsize=(8,3))
  ax=[ fig.add_subplot(111) for i in [1,2] ] # Add subplot for each scenario
  drivers = ['NEIGH', 'TRAIN', 'TRAVEL', 'LU']
  ind = np.arange(len(all_metrices))    # the x locations for the groups
  width = 0.13 # width of a bar
  space = 0.02 # space between case studies
  alpha = {1:0.9,2:0.6}
  parameterSets = calibrate.getParameterConfigurations()
  plt.ylabel('parameters')
  plt.ylim([0,1.1])
  plt.xlim([-0.5,10])
  fig.text(0.42, 0.02,"goal function")
  plt.xticks(ind, goal_functions)

  # Create a dictionairy to store lists with parameters. Lenght of dict = no of drivers
  # One list = one parameter values for 5 goal functions 3 case studies, a scenario (scenarios are sepearetly)
  p_dict = {}
  
  # Fill the data:
  for scenario in [1,2]:
    p_dict[scenario]={}
    for driver in range(len(drivers)):
      a_list=[]
      for metric in all_metrices:
        for country in case_studies:    
          # get the index of the best parameter set
          calibratedIndex = calibrate.getCalibratedIndeks(metric,scenario,country)
          a_list.append(parameterSets[calibratedIndex][driver])
      p_dict[scenario][driver]=a_list

  for driver in range(len(drivers)):
    for scenario in [1,2]:
      print('scenario: ',scenario,'driver', drivers[driver])
      np.set_printoptions(precision=2, linewidth=100)
      print(np.array(p_dict[scenario][driver]))
  #Assign positions of bars for each scenario:
  positions = {
    1:np.array([[i - (2.5*width+space),i - 0.5*width,i + (1.5*width+space)] for i in ind ]).flatten(),
    2:np.array([[i - (1.5*width+space),i + 0.5*width,i + (2.5*width+space)] for i in ind ]).flatten()}
  # Now plot the bars. For each scenario, the bars are plotted as a different ax
  for scenario in [1,2]:
    bottom=0
    for driver in range(len(drivers)):
      label = {1:drivers[driver], 2:None}
      ax[scenario-1].bar(
        positions[scenario],
        p_dict[scenario][driver],
        width=width,
        bottom=bottom,
        color=driverColors[driver],
        alpha=alpha[scenario],
        label=label[scenario])
      bottom = bottom + np.array(p_dict[scenario][driver])

  # Draw lines dividing scenario bars and add annotation with the country symbol
  for xtick in [0,1,2,3,4]:
    for c,country in enumerate(case_studies):
      p = xtick-2*width-space+c*(2*width+space)
      plt.axvline(p, alpha=0.5,c='white',linestyle='--', linewidth=0.5)
      plt.annotate(country,xy=(p,1.01), rotation=0, color="darkviolet",ha='center',weight='bold')

  #Create a legend
  handles, labels = ax[0].get_legend_handles_labels()
  patch = handles[0][0]
  # plot the driver in the right top corner. Reverse the order to adjust tp the plot:
  leg1 = ax[0].legend(
    handles[::-1],
    labels[::-1],
    loc=1)
  plt.gca().add_artist(leg1)
  # Plot two patches with different opacity to show the difference between scenario 1 and 2:                   
  leg2 = plt.legend(
    handles=[patch,patch],
    labels=['scenario 1','scenario 2'],
    bbox_to_anchor=(0, 1.05, 1, .102),
    loc='lower center',
    ncol=2,
    mode="expand",
    bbox_transform=ax[0].transAxes,
    borderaxespad=0.)
  leg2.get_frame().set_edgecolor('darkviolet')
  leg2.get_frame().set_linewidth(0.50)
  for i,lh in enumerate(leg2.legendHandles): 
    lh.set_alpha(alpha[i+1])
  
  # Set the name and clear the directory if needed

  setNameClearSave('plotParameters', scenario=None)

def plotParametersMultiobjective(): 
  """
  Plot bars presenting parameters (0-1, y axis) for 5 goal funcstions (x axis)
  There are 5 metrics, 3 case studies, 2 scenarios --> 15 bars
  Ignore the warning
  """
 
  # Create figure
  fig = plt.figure(figsize=(8,3))
  ax=[ fig.add_subplot(111) for i in [1,2] ] # Add subplot for each scenario
  drivers = ['NEIGH', 'TRAIN', 'TRAVEL', 'LU']
  ind = np.arange(len(metricNames))    # the x locations for the groups
  width = 0.13 # width of a bar
  space = 0.02 # space between case studies
  alpha = {1:0.9,2:0.6}
  parameterSets = calibrate.getParameterConfigurations()
  plt.ylabel('parameters')
  plt.ylim([0,1.1])
  plt.xlim([-0.5,5.25])
  fig.text(0.42, 0.02,"goal function")
  plt.xticks(ind, goal_functions)

  # Create a dictionairy to store lists with parameters. Lenght of dict = no of drivers
  # One list = one parameter values for 5 goal functions 3 case studies, a scenario (scenarios are sepearetly)
  p_dict = {}
  
  # Fill the data:
  for scenario in [1,2]:
    p_dict[scenario]={}
    for driver in range(len(drivers)):
      a_list=[]
      for metric in metricNames:  #multiobjective
        for country in case_studies:    
          # get the index of the best parameter set
          calibratedIndex = calibrate.getMultiObjectiveIndex(metric,0.5,0.5,scenario,country) # multiobjective
          a_list.append(parameterSets[calibratedIndex][driver])
      p_dict[scenario][driver]=a_list

  for driver in range(len(drivers)):
    for scenario in [1,2]:
      print('scenario: ',scenario,'driver', drivers[driver])
      np.set_printoptions(precision=2, linewidth=100)
      print(np.array(p_dict[scenario][driver]))
  #Assign positions of bars for each scenario:
  positions = {
    1:np.array([[i - (2.5*width+space),i - 0.5*width,i + (1.5*width+space)] for i in ind ]).flatten(),
    2:np.array([[i - (1.5*width+space),i + 0.5*width,i + (2.5*width+space)] for i in ind ]).flatten()}
  # Now plot the bars. For each scenario, the bars are plotted as a different ax
  for scenario in [1,2]:
    bottom=0
    for driver in range(len(drivers)):
      label = {1:drivers[driver], 2:None}
      ax[scenario-1].bar(
        positions[scenario],
        p_dict[scenario][driver],
        width=width,
        bottom=bottom,
        color=driverColors[driver],
        alpha=alpha[scenario],
        label=label[scenario])
      bottom = bottom + np.array(p_dict[scenario][driver])

  # Draw lines dividing scenario bars and add annotation with the country symbol
  for xtick in [0,1,2,3]:
    for c,country in enumerate(case_studies):
      p = xtick-2*width-space+c*(2*width+space)
      plt.axvline(p, alpha=0.5,c='white',linestyle='--', linewidth=0.5)
      plt.annotate(country,xy=(p,1.01), rotation=0, color="darkviolet",ha='center',weight='bold')

  #Create a legend
  handles, labels = ax[0].get_legend_handles_labels()
  patch = handles[0][0]
  # plot the driver in the right top corner. Reverse the order to adjust tp the plot:
  leg1 = ax[0].legend(
    handles[::-1],
    labels[::-1],
    loc=1)
  plt.gca().add_artist(leg1)
  # Plot two patches with different opacity to show the difference between scenario 1 and 2:                   
  leg2 = plt.legend(
    handles=[patch,patch],
    labels=['scenario 1','scenario 2'],
    bbox_to_anchor=(0, 1.05, 1, .102),
    loc='lower center',
    ncol=2,
    mode="expand",
    bbox_transform=ax[0].transAxes,
    borderaxespad=0.)
  leg2.get_frame().set_edgecolor('darkviolet')
  leg2.get_frame().set_linewidth(0.50)
  for i,lh in enumerate(leg2.legendHandles): 
    lh.set_alpha(alpha[i+1])
  
  # Set the name and clear the directory if needed
  setNameClearSave('plotParameters_multiobjective', scenario=None)

def plotObsAndMod():
  """
  # Plots two subplots for each metric
  # Plot the observed values of metrics in one subplot
  # and the modelled metric values for the calibrated parameters on the other subplot
  # All landscape metrics = 4 plots, 2 subplots each, stacked vertically
  # For FDI and WFDI plots values for zones on the diagonal of the case study area (zones: 0,5,10,15)
  """
  ## 1. Create the figure
  fig = plt.figure(figsize=(10,12)) # Figure size set to give 16 cm of width
  fig.align_ylabels()
  
  # Set linestyle for each scenario
  linestyle={1:'o-',2:'o--'}
  ## 2. Get results for countries, scenarios and all metrics (with kappa)
  parameterSets = calibrate.getParameterConfigurations()
  
  i=1
  for m,metric in enumerate(metricNames):
    axs1 = plt.subplot(4,2,i)
    axs2 = plt.subplot(4,2,i+1, sharey=axs1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # Create list to store limits:
    limits_min= []
    limits_max= []
    limits_mean = []
    obs_values_all=np.array([])
    for c,country in enumerate(case_studies):
      zonesObserved = calibrate.getObservedArray(metric,case=country)
      # Find the maximum and minimum value

      obs_values = np.array([zonesObserved[y,0,1] for y in range(len(obsTimeSteps))])
      obs_values = obs_values[~np.isnan(obs_values)].flatten()
      limits_min.append(np.amin(obs_values))
      limits_max.append(np.amax(obs_values))
      limits_mean.append(np.mean(obs_values))
      obs_values_all = np.append(obs_values_all,obs_values)
        
      for scenario in [1,2]:
        zonesModelled = calibrate.getModelledArray(metric,scenario,case=country)
        indexCalibratedSet = calibrate.getCalibratedIndeks(metric,scenario,case=country)

        ## 3. Plot the values        
        # Select the zones for plottong. Some metric are calculted for 16 zones. Other for one zone only:
        zonesNumber = len(zonesModelled[0][0][1])
        if zonesNumber > 1:
          selected_zones = range(zonesNumber)#[0,5,10,15]
        else:
          selected_zones = range(zonesNumber)
          
        # Loop the zones:
        for z in selected_zones:
          # Plot observed values. Create and array and fill with values for each observed time step:
          metricValues = []
          for year in range(len(obsTimeSteps)):
            metricValues.append(zonesObserved[year][0][1][z][0])
          axs1.plot(
            observedYears,
            metricValues,
            linestyle[scenario],
            linewidth = 0.7,
            markersize=1,
            label = country+str(scenario),
            c=countryColors[country])
          
          # Plot modelled values. Create and array and fill with values for each time step:
          modelledValues = []
          for year in range(len(zonesModelled[:,0])):
            modelledValues.append(zonesModelled[year,indexCalibratedSet,1][z][0])
          axs2.plot(
            np.arange(observedYears[0],observedYears[-1:][0]+1),
            modelledValues,
            linestyle[scenario],
            linewidth = 0.7,
            markersize=1,
            label = country+str(scenario),
            c=countryColors[country])
      # Get the minimum and maximum values:
      if country == case_studies[-1]:
        a_min = np.amin(limits_min)
        a_max = np.amax(limits_max)
        # Plot lines on the observed plot
        step = (a_max-a_min)/5
        for y in range(6):  
          axs1.axhline(a_min+y*step, ls='--',c='black', alpha = 0.1, lw=0.5)
        # Plot marks showin mean values for all years:
        axs1.scatter(
          [observedYears[-1]+2,observedYears[-1]+2,observedYears[-1]+2],
          limits_mean,
          c = [colors.to_rgba(countryColors[c]) for c in case_studies])
        bp = axs1.boxplot(obs_values_all, positions =[observedYears[-1]+2], widths = 2)
        for flier in bp['fliers']:
          flier.set(marker='o', color='black', alpha=0.5, markersize=0.5)
          
      axs1.set_xlim([observedYears[0]-2,observedYears[-1]+4])

      # Adjust labels of the 'observed' and 'modelled' subplots
      axs1.set_title('Observed')
      axs2.set_title(goal_functions[m])
      
      axs1.set_xticks(observedYears)
      axs1.set_xticklabels(observedYears)
      axs1.set_ylabel('metric '+metric.upper())
      if metric=='kappa':
        axs1.set_ylabel(metric[0].upper())
      axs1.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
      
      xYears = np.arange(1990,2018+1, step=4)
      axs2.set_xticks(xYears)
      axs2.set_ylabel(None)
      axs2.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
      
      
    i+=2

  # Move the legend outside of the plot box. Put it below, with the zones in two rows.:
  leg=axs2.legend(
    bbox_to_anchor=(-1-0.2,-0.35, 2.2, 0.2),
    loc="upper left",
    mode = "expand",
    bbox_transform=axs2.transAxes,
    ncol=6,
    borderaxespad=0.)

  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  
  # Set the name and clear the directory if needed
  setNameClearSave('obs_and_mod')

def plotValidation(errors_or_kappas):  # Done
  """  
  Plot bars presenting validation metrics values (y axis) for 5 goal funcstions (x axis)
  There are 5 validation metrics showing errors or disagreement: 'errors'
  There are 2 validation metrics showing Kappa statistics: 'kappas'

  errors_or_kappas in ['errors','kappas']
  """
  rows = {
    'errors':[0,1,2,3,6],
    'kappas': [4,5]
    }
  
  # Get the array with validation results for selected validation metrics:
  results = calibrate.getValidationResults()[rows[errors_or_kappas],:]
  v_m_subset= [validation_metrices[i] for i in rows[errors_or_kappas]]
  # Create figure
  height = len(rows[errors_or_kappas])*1.1
  fig, axs = plt.subplots(len(rows[errors_or_kappas]),1,figsize=(7.14,height),sharex=True)
  ind = np.arange(len(all_metrices))    # the x locations for the groups
  n = 3*2 # 3 countries, two scenarios
  width = 0.1 # width of a bar
  space = 0.02 # space between case studies
  alpha = {1:0.9,2:0.6}
  plt.xlabel('goal function')
  plt.xticks(ind, [goal_functions[i] for i in range(len(all_metrices))])
  fig.align_ylabels()
  plt.subplots_adjust(wspace=0.1, hspace=0.25)
  
  #Assign positions of bars for each of case studies and scenarios:
  positions = np.array(
    [[i - (2.5*width+space),
      i - (1.5*width+space),
      i - 0.5*width,
      i + 0.5*width,
      i + (1.5*width+space),
      i + (2.5*width+space)] for i in ind ]).flatten()
  # Assign y labels:
  ylabel = [ metric_units[i] for i in rows[errors_or_kappas]]
  # Prepare colors:
  bar_c=[]
  for case in case_studies:
    for s in [1,2]:
      a_color = colors.to_rgba(countryColors[case])[:-1]+(alpha[s],)
      bar_c.append(a_color)
  # Prepare list for bars and labels
  bars = []
  labels = [x+str(y) for x in case_studies for y in [1,2]]
  # Now plot the bars for the selected validation merics. For each metric, the bars are plotted as a different ax
  for v_m,v_metric in enumerate(v_m_subset):
    axs[v_m].set_title(v_m_subset[v_m], pad=2)
    axs[v_m].set_ylabel(ylabel[v_m])
    axs[v_m].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
    bar=axs[v_m].bar(
        positions,
        results[v_m],
        color = bar_c,
        width=width)
    if v_m == 0:
      bars.append(bar)
    # If we are working with Kappas, apply the limits:
    if v_metric == 'K':
      axs[v_m].set_ylim([0.45,0.9])
    if v_metric == 'Ks':
      axs[v_m].set_ylim([0.45,0.9])
    # Draw lines dividing scenario bars and add annotation with the country symbol
    for xtick in ind:
      for c,country in enumerate(case_studies):
        p = xtick-2*width-space+c*(2*width+space)
        axs[v_m].axvline(p, alpha=0.5,c='white',linestyle='--', linewidth=0.5)
        
  # Create a legend:
  leg = axs[0].legend(
    [ bar for bar in bars[0]] ,
    labels,
    bbox_to_anchor=(0., 1.3, 1, .102),
    loc='lower center',
    ncol=6,
    mode="expand",
    bbox_transform=axs[0].transAxes,
    borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)  
 
  # Set the name and clear the directory if needed
  setNameClearSave('plotValidation_'+errors_or_kappas, scenario=None)
#plotValidation('kappas')

def plotMultiobjectiveImprovementToLocationalMetric(weights,loc_metric, positive=False):
  """
  # loc_metric in ['K','Ks,'A'] (Kappa, Kappa Simulation and Allocation Disagreement)
  # Plot the improvement obtained using of multi-objective goal function used in calibration.
  # 4 landscape metrics used in goal fuunctions => 4 subplots
  # weights = [w_RMSE, w_Kappa] <- importance of each goal function in multiobjective optimisation
  # positive = whether only positive values of RMSE are to be plotted
  """
  locational_metric = {
    'K':4,
    'Ks':5,
    'A':6}
 
  ## 1. Get the data. 
  results = calibrate.getValidationResults() # array 7x30
  results_multio = calibrate.getValidationResults_multiobjective(weights)
  ## 2. Calculate the improvement between the metrics values calculated using kappa goal function and multi-o
  # Take only the last n columns from single-objective results,
  # as it stores values for goal function Kappa 
  locational_gf = results[:,-len(cases):]
  # Broadcast it to the shape of results
  locational_array = np.concatenate((locational_gf,locational_gf,locational_gf,locational_gf),axis=1)
  # Get the improvement comapring to single-objective function
  improvement = ((locational_array-results_multio)/locational_array)*100 #%
  # Rows 4 and 5 contain accuracy metrics, so the value should increase. The others rows contains errors.
  # Change the sign in accuracy rows:
  improvement[[4,5],:] = -improvement[[4,5],:]
  # If want to plot the the positive values of RMSE only, remove the negative values:
  if positive is True:
    improvement[0:4][improvement[0:4]<0] = np.NaN
  
  # Find benchmark for locational metric (coefficient of variation CV for each case):
  benchmark = np.empty((len(validation_metrices),len(cases),2)) # cases x top and bottom benchmarks
  # Cretae array to store benchmark
  saveArrayAsExcel([
    ['File containing Coefficient of Variation for each country and scenario'],
    ['CV = std/mean'],
    ['CV calculated for validation metrics values obtained for all single-objective goal functions n=5']],
                   'coefficient_of_variation',sheet_name = 'CV file')
  # For each validation results, find benchmark for single-objective goal function
  for vm, v_metric in enumerate(validation_metrices):
    # For each country and scenario, calculate mean and sd of locational values obtained for 5 goal functions
    for i, (c,s) in enumerate(cases):
      # Get validation results for all goal functions but one country and scenario
      val_r = results[vm,i:results.shape[1]:len(cases)]
      # Calculate coefficient of variation
      cv = np.std(val_r)/np.mean(val_r) * 100 # [%]
      # Get the benchmark
      benchmark[vm][i] = [-cv,cv ] # in %
    # create a dataframe to store it in excle file
    df = pd.DataFrame(benchmark[vm], index = cases, columns = ['-CV','CV'])
    appendArrayAsExcel(df, 'coefficient_of_variation', sheet_name = v_metric)
    
  ## 3. Prepare the plot!
  # Create a figure
  fig, axs = plt.subplots(2,2, figsize=(7.14,7.14)) # 16 cm of width
  # Add y label
  fig.text(0.04, 0.5, loc_metric+' improvement [%]', va='center', rotation='vertical')
  # Add x label
  fig.text(0.4, 0.04, 'RMSE improvement [%]', va='center')
  # Arrange subplots:
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  # Assign markers
  marker = {
    'cilp':'s', # square
    'fdi':'^',  # triangle
    'wfdi':'X', # cross
    'pd':'o'}   # circle
  # Assign colors:
  alpha = {1:1, 2:0.6}
  c_color=[]
  for case in case_studies:
    for s in [1,2]:
      a_color = colors.to_rgba(countryColors[case])[:-1]+(alpha[s],)
      c_color.append(a_color)
  # Make a mask for plotting for each case (scenario):
  n = 3*2
  mask = [i for i in range(n)]
  # Plot improvements on four subplots, each with 4 iteration of axes (as 4 multi-o goal functions)
  i = 0
  j = 0
  for m_v,metric_v in enumerate(metricNames):
    # Add background color:
    axs[i,j].set(facecolor='#DCDCDC')
    # Draw lines for zeros
    axs[i,j].axvline(x=0, alpha=0.7, c='black', lw=1)
    axs[i,j].axhline(y=0, alpha=0.7, c='black', lw=1)
    axs[i,j].set_title(metric_v.upper())
    # If plotting all values, set y axis into log scale
    if positive is False:
      axs[i,j].set_yscale('symlog')
    y_min=0
    # plot seperately values for each goal function. Get an array for all cases, one goal function:
    cols = [np.array(mask)+i*n for i in range(len(metricNames))]
    
    for gf,col in enumerate(cols):
      # Compare RMSE improvement results to benchmark and the locational to be smaller than one standard deviation:
      b_test1 = improvement[m_v,col]>benchmark[m_v,:,1]
      b_test2 = improvement[locational_metric[loc_metric],col]>benchmark[locational_metric[loc_metric],:,0]
      b_test = [b1 and b2 for b1,b2 in zip(b_test1,b_test2)]
      b_facecolor = []
      # If the marker crosses the benchmark (CV), fill it with color
      for bi,acolor in enumerate(c_color):
        if b_test[bi] == True:
          b_facecolor.append('white')
        else:
          b_facecolor.append('none')
      # Find minimum y:
      y_min_a = np.array([improvement[m_v,col],improvement[locational_metric[loc_metric],col]])
      y_min_a = y_min_a[:,~np.isnan(y_min_a).any(axis=0)] 
      if y_min_a[1].any() and np.amin(y_min_a[1])<y_min:
        y_min=np.amin(y_min_a[1])
      # Now, for each goal function (col), assign markers depending on the metric used in the goal function
      axs[i,j].scatter(
        improvement[m_v,col],
        improvement[locational_metric[loc_metric],col],
        #s=20,
        facecolors=[b for b in b_facecolor],
        edgecolors=c_color,
        marker = marker[metricNames[gf]])
    # If showing only positive vaues, limit the plot
    if positive is True:
      axs[i,j].set_xlim([-5,100])
      axs[i,j].set_ylim([y_min*1.1,-y_min*0.1])
      
    # Plot lines and labels showing benchmarks for landscape metrics only dor all cases
    #if positive is False:
    # For better clarity plot ONLY the max benchmark
    for c,case in enumerate(cases):      
      # Plot low and upper benchmark:
      boundries = [0,1]
      for b in boundries:
        # Plot the line
        axs[i,j].axvline(x=benchmark[m_v,c,b], ls='--',color = c_color[c], lw= 0.75)
        # Add the label
        s = '{:.2f}'.format(benchmark[m_v,c,b])
        
        # plot x banchmark value only for the max benchmark:
        if benchmark[m_v,c,1] == np.amax(benchmark[m_v,:,1]) and positive is False:
          delta=0.02
          if m_v=='cilp': # the frst plot can't het the ylim correct
            delta = 0.5
          axs[i,j].text(x=benchmark[m_v,c,b],
                        y=axs[i,j].get_ylim()[1]+delta,
                        s=s,
                        fontsize=6,
                        color = c_color[c],
                        ha = 'center')
          
    # Plot lines and labels showing benchmarks for locational metric
    s_delta = [0.01,-0.1,-0.07,-0.04,-0.25,-0.025]
    for c,case in enumerate(cases):
      # Plot the line
      axs[i,j].axhline(y=benchmark[locational_metric[loc_metric],c,0], ls='--',color = c_color[c], lw= 0.75)
      # Add the label
      if positive is False:
        s = '{:.2f}'.format(benchmark[locational_metric[loc_metric],c,0])
        axs[i,j].text(x=axs[i,j].get_xlim()[1]+0.2,
                      y=benchmark[locational_metric[loc_metric],c,0]+s_delta[c],
                      s=s,
                      fontsize=6,
                      color = c_color[c])
    j=+1
    if m_v==1:
      i=1
      j=0

  ## 4. Create the legends from scratch
  # First, for patches representing the meaning of colors:
  # Make labels patches:
  labels = [x+str(y) for x in case_studies for y in [1,2]]
  patches = [ mpatches.Patch(color=c_color[j], label="{l}".format(l=labels[j]) ) for j in range(len(labels)) ]
  # Make the first legend:
  leg1 = axs[0,0].legend(
    patches,
    labels,
    bbox_to_anchor=(0., 1.25, 2+0.2, .102),
    loc='lower center',
    ncol=len(patches),
    mode="expand",
    borderaxespad=0.,
    bbox_transform=axs[0,0].transAxes,
    fontsize=6)
  axs[0,0].add_artist(leg1)
  
  # Second, for markers used in plot
  # Make marker objects:
  markers = []
  for m, metric in enumerate(metricNames):
    markers.append(mlines.Line2D([], [],
                                 marker=marker[metric],
                                 linestyle='none',
                                 mec = c_color[0],
                                 mfc = 'none',
                                 markersize=6,
                                 label=mo_goal_functions[m]))

  # Add second legend
  leg2 = axs[0,0].legend(
    handles = markers,
    bbox_to_anchor=(0., 1.15, 2+0.2, .102),
    loc='lower center',
    ncol=len(metricNames),
    mode="expand",
    borderaxespad=0.,
    bbox_transform=axs[0,0].transAxes,
    fontsize=6)

  # Adjust borders of both legends:
  for leg in [leg1,leg2]:
    leg.get_frame().set_edgecolor('darkviolet')
    leg.get_frame().set_linewidth(0.50)
    
  # Set the name and clear the directory if needed
  ending = {True: 'positive_only',False:'all_values'}
  name = 'multiobjective_compared_to_'+loc_metric+'_'+ending[positive]
  setNameClearSave(name)
#plotMultiobjectiveImprovementToLocationalMetric([0.5,0.5],'K', positive=True)
#plotMultiobjectiveImprovementToLocationalMetric([0.5,0.5],'K', positive=False)

def plotNonDominatedSolutions(metrics):
  '''
  Find non-dominated combinations of metric values (for each parameter set)
  Plot 6 suplots, for each case study and scenario
  in each case study x and y axis show values of two metrics
  thrid metric value is presented with shape
  '''
  # Create the figure
  fig, axs = plt.subplots(3,2, figsize=(4,5), sharex=True, sharey=True)
  # Add y label
  fig.text(0, 0.5, 'normalized value of metric '+metrics[0].upper(), va='center', rotation='vertical')
  # Add x label
  fig.text(0.5, 0, 'normalized value of metric '+metrics[1].upper(), ha='center')
  
  # Get normalized (0-1) results of validation metrics
  results = {
    'calibration':createNormalizedResultDict(metrics,'calibration'),
    'validation':createNormalizedResultDict(metrics,'validation')}
  # Set the number of parameter combinations
  iterations = len(calibrate.getParameterConfigurations())
  # Set the subplot indices:
  i=j=0
  i_index = lambda x,y: x+1 if y==1 else x
  j_index = lambda x: x+1 if x==0 else 0
  
  # Loop each simulation
  for country, scenario in cases:
    # Assign the number of non-dominated solutions
    count=0
    # Select non-dominated calibration solutions
    non_dominated = is_pareto_efficient_simple(results['calibration'][country][scenario])
    # Get validation values
    v = results['validation'][country][scenario]
    # Subset non-dominated calibration results
    r_nd = v[non_dominated]
    # Find the maximum values
    max0, max1, max2 = r_nd.max(axis=0)
    # Plot a scatter plot:
    axs[i,j].scatter(
      r_nd[:,0],
      r_nd[:,1],
      s=20*np.power(r_nd[:,2],3),
      facecolors=countryColors[country],
      edgecolors='none',
      alpha=0.3,
      label = metrics[2])
    # Plot the point with the maximum size again
    s_r_nd = r_nd[r_nd[:,2]==max2]
    axs[i,j].scatter(
      s_r_nd[:,0],
      s_r_nd[:,1],
      s=20*np.power(s_r_nd[:,2],3),
      facecolors='none',
      edgecolors=countryColors[country],
      alpha=0.7,
      label = f'Highest {metrics[2]}')
    # Set the left border
    axs[i,j].set_xlim(left=-0.05)
    axs[i,j].set_ylim(bottom=-0.05)
    
    # Print the number of non-dominated solutions in the bottom left corner
    axs[i,j].text(0,0,'n='+str(len(r_nd)))
    # Get the indices of subplots:
    i=i_index(i,j)
    j=j_index(j)
    
  axs[0,0].legend(fontsize=6)
  # Save plot and clear
  setNameClearSave('2_non-dominated-scatter_'+'_'.join(metrics))

plotParameters()
