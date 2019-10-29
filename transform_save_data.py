# RESOLVE: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory.

# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import metrics
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt

#### Script to read in the metrics saved as the result of the LU_urb.py script.
#### Metrics are transformed into an array

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps=parameters.getNrTimesteps()
numberOfIterations = parameters.getNumberofIterations(parameters.getSuitFactorDict(), parameters.getParametersforCalibration())

iterations = range(1, numberOfIterations+1, 1)
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsSampleNumbers = [1] #range(1,20+1,1) <- for stochastic model
obsTimeSteps = [10,16,22,28]

# Path to the folder with the metrics stored
country = parameters.getCountryName()
resultFolder = os.path.join(os.getcwd(),'results',country)
output_mainfolder = os.path.join(resultFolder, "metrics")

#################
### FUNCTIONS ###
#################

def openPickledSamplesAndTimestepsAsNumpyArray(basename,iterations,timesteps, \
                                               obs=False):
  output=[]
  
  for timestep in timesteps:
    allIterations=[]
    
    for i in iterations:
      # Read in the parameters
      pName = 'parameters_iteration_' + str(i) + '.obj'
      pFileName = os.path.join(resultFolder, str(i), pName)
      filehandler = open(pFileName, 'rb') 
      pData = pickle.load(filehandler)
      pArray = np.array(pData, ndmin=1)
          
      # If we are working with the observed data (CLC data):
      if obs:
        name = generateNameT(basename, timestep)
        fileName = os.path.join('observations', country, 'realizations', str(i), name)
        data = metrics.map2Array(fileName, os.path.join('input_data', country, 'sampPoint.col'))
        
      # If we are working with the observed values:  
      else:
        theName = basename + str(timestep) + '.obj'
        fileName = os.path.join(resultFolder, str(i), theName)
        filehandler = open(fileName, 'rb') 
        data = pickle.load(filehandler)
        '''# Keep these lines for the later use:
        if it is a dictionary, get the sugar cane parameters (lu type 6)
        if type(data) == dict:
          data = data.get(1)
        filehandler.close()'''
      # if the loaded data was not yet an array, make it into one
      # minimum number of dimensions is one, to prevent a zero-dimension array
      # (not an array at all)
      array = np.array(data, ndmin=1)
      # add an extra dimension that would normally be y, if the data was a map
      # so that Derek's plot functions can be used
      array = array.reshape(len(array),1)
      allIterations.append([pArray,array])
    output.append(allIterations)
  outputAsArray = np.array(output)
  return outputAsArray

def saveSamplesAndTimestepsAsNumpyArray(basename, iterations, timesteps, obs=False):
  # Convert the output of the model into arrays
  output = openPickledSamplesAndTimestepsAsNumpyArray(basename, iterations, timesteps, obs)
  
  # Check if the directory exists. If not, create.
  if not os.path.isdir(output_mainfolder):
      os.mkdir(output_mainfolder)
      
  # Set the name of the file
  if obs:
    fileName = os.path.join(output_mainfolder, basename + '_obs')
  else:
    fileName = os.path.join(output_mainfolder, basename)

  # Clear the directory if needed
  if os.path.exists(fileName + '.npy'):
      os.remove(fileName + '.npy')

  # Save the data  
  np.save(fileName, output)
  

#################################
### SAVE OUTPUTS OF THE MODEL ###
#################################
  
# now save all outputs of the model as one array per variable
# so that we can delete all number folders
 
variables = parameters.getSumStats()
print("Save modelled and observed metrics: ", variables)

# Save for the modelled and observed metrics:
for aVariable in variables:
  saveSamplesAndTimestepsAsNumpyArray(aVariable, iterations, timeSteps)
  saveSamplesAndTimestepsAsNumpyArray(aVariable, obsSampleNumbers,obsTimeSteps, True)  

######################################
### VISUALIZE OUTPUTS OF THE MODEL ###
######################################
  
print("Parameter values are stored in 3 dimensional array [timestep, iteration, metric]")
print("# timestep: year, \n# iteration: set of parameters used, \n# metric: array of values \
of the selected metric for each zone for each set of parameters seperately")

######################################
# All histograms together
for aVariable in variables:
  h = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))
  #print('e.g. timestep 1 (year 1991)')
  #print(h[1,:,:])
  for timeStep in timeSteps:
    bins = numpy.linspace(-100, 100, 100)
    hTitle = "Histogram for metric "+aVariable+" with 'auto' bins"
    plt.title(hTitle)
    plt.xlabel('Metric:' + aVariable)
    plt.ylabel("Frequency")
    plt.hist(h[timeStep-1][0][1], bins='auto', alpha=0.5, label='timestep: '+str(timeStep))
    plt.legend(loc='upper right')
    
  # Set the name and clear the directory if needed
  hNameAll = "histogram_"+ aVariable +".png"
  if os.path.exists(hNameAll):
      os.remove(hNameAll)
  # Save
  plt.savefig(os.path.join(output_mainfolder,hNameAll))
  plt.clf()

for aVariable in variables:
  test = np.load(os.path.join("results", country, 'metrics', aVariable + '_obs.npy'))
  for timeStep in range(1,len(obsTimeSteps)+1):
    hTitle = "Histogram for metric "+aVariable+" with 'auto' bins"
    plt.title(hTitle)
    plt.xlabel('Metric:' + aVariable)
    plt.ylabel("Frequency")
    plt.hist(test[timeStep-1][0][1], bins='auto', alpha=0.5, label='timestep: '+str(obsTimeSteps[timeStep-1]))
    plt.legend(loc='upper right')

  # Set the name and clear the directory if needed
  hNameAllObs = "histogram_"+ aVariable +"_obs.png"
  if os.path.exists(hNameAllObs):
      os.remove(hNameAllObs)   
  # Save
  plt.savefig(os.path.join(output_mainfolder,hNameAllObs))
  plt.clf()


######################################
# One histogram of metric values for different parameters per zone per timestep

for aVariable in variables:
  zonesModelled = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))

  for timeStep in range(0,len(zonesModelled)): # Loop data for each time step
    # Prepare the main plot for each timeStep
    fig, axes = plt.subplots(nrows=4, ncols=4) # Depending on number of zones!
    hTitle = "Histogram of metric "+aVariable+" for each zone in timestep: " + str(timeStep + 1)
    fig.suptitle(hTitle)
    fig.subplots_adjust(hspace=1)
    
    for zone in range(0, len(zonesModelled[timeStep][0][1])): # Loop array to extraxt data for each zone
      # Prepare data for each zone to be plotted in each subplot
      zoneMetric = []
      metrics_for_parameters = []

      for i in range(0,len(zonesModelled[timeStep])): # Loop array to gest the metric for the time step in given zone
        metrics_for_parameters.append(zonesModelled[timeStep][i][1][zone][0]) # [0] gives the raw number

      # Plot subplots
      axes.flatten()[zone].hist(metrics_for_parameters, bins = 'auto', )
      axes.flatten()[zone].set(title=zone+1)

      # bins=len(np.unique(data.data.T[0]))//2 MAYBE USE LATER

    # Set the name and clear the directory if needed
    hName = 'Histogram_' + aVariable + "_timestep_"+ str(timeStep+1) + ".png"
    if os.path.exists(hName):
        os.remove(hName)

    # Save plot and clear    
    plt.savefig(os.path.join(output_mainfolder,hName))
    plt.clf()

print('Histograms for each zone and ech time step plotted.')  

  
  
  


