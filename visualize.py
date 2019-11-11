# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

# Get metrics
metricNames = parameters.getSumStats()

# Get zones
zones = 16 # Maybe to parameters??

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps = parameters.getNrTimesteps()
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsTimeSteps = [10] #parameters.getObsTimesteps

# Path to the folder with the metrics stored
country = parameters.getCountryName()
resultFolder = os.path.join(os.getcwd(),'results',country, 'metrics')

#################
### FUNCTIONS ###
#################

def histogramsModelledMetrics(theMetrics):
  #('number of time steps: ',len(zonesModelled))
  #('number of parameter configurations: ',len(zonesModelled[0]))                     
  #('number of zones: ',len(zonesModelled[0][0][1]))

  for aVariable in theMetrics:
    zonesModelled = np.load(os.path.join("results", country, 'metrics', aVariable + '.npy'))
    
    for timeStep in timeSteps: # Loop data for each time step
      # Prepare the main plot for each timeStep
      fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey = True) # Depending on number of zones!
      hTitle = "Histogram of modelled metric "+aVariable+" for each zone in timestep: " + str(timeStep)
      fig.suptitle(hTitle)
      fig.subplots_adjust(hspace=1)
      
      for zone in range(0, len(zonesModelled[timeStep-1][0][1])): # Loop array to extraxt data for each zone
        # Prepare data for each zone to be plotted in each subplot
        metrics_for_histogram = []

        for i in range(0,len(zonesModelled[timeStep-1])): # Loop array to get metrics for each parameter configuration
          metrics_for_histogram.append(zonesModelled[timeStep-1][i][1][zone][0]) # [0] gives the raw number
          
        # Plot subplots
        axes.flatten()[zone].hist(metrics_for_histogram, bins = 'auto', )
        axes.flatten()[zone].set(title=zone+1)
        axes.flatten()[zone].ticklabel_format(axis='x', style='sci', scilimits=(-4,4))

        # bins=len(np.unique(data.data.T[0]))//2 MAYBE USE LATER

      # Set the name and clear the directory if needed
      hName = 'Histogram_' + aVariable + "_modelled_timestep_"+ str(timeStep) + ".png"
      if os.path.exists(hName):
          os.remove(hName)

      # Save plot and clear    
      plt.savefig(os.path.join(resultFolder,hName))
      plt.clf()

      # Close all open figures
      plt.close('all')

def transformArray(theArray):
  # Create an array storing absolute difference values in the shape of the zones
  numberZones = len(theArray[0][0])
  parameterSets = len(theArray[0])
  # Create array
  n = int(np.sqrt(numberZones))
  newArray = np.zeros((parameterSets,n,n))
  # Fill with values
  for pSet in range(0,parameterSets):
    for zone in range(0,numberZones):
      i = np.floor_divide(zone,4)
      j = np.remainder(zone,4)
      newArray[pSet][i][j] = theArray[0][pSet][zone]
  return newArray

def plotAbsoluteDifference(metricNames, obsTimeStep):
  for aMetric in metricNames:
    differenceArray = np.load(os.path.join(resultFolder, aMetric + '_diff.npy'))
    array = transformArray(differenceArray)
    parameterSets = len(differenceArray[0])
    NoRows = int(np.ceil(parameterSets/4))
        
    fig = plt.figure(figsize=(12, 40))
    
    for pSet in range(0,parameterSets):
      plt.subplot(NoRows,4,pSet+1)
      plt.imshow(array[pSet], cmap='Purples')
      plt.xlabel('Parameter set: '+ str(pSet+1))
      plt.axis('off')
      plt.title('Parameter set: '+ str(pSet+1))

    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.9, wspace=0.2)
    cax = plt.axes([0.125, 0.10, 0.675, 0.025])
    plt.colorbar(cax = cax,orientation='horizontal')
    
    # Set the name and clear the directory if needed
    pName = 'Difference_' + aMetric + "_timestep_"+ str(obsTimeStep) + ".png"
    pPath = os.path.join(resultFolder, pName)
    if os.path.exists(pPath):
        os.remove(pPath)
        
    # Save plot and clear    
    plt.savefig(os.path.join(resultFolder,pPath))
    plt.close('all')


######################################
### VISUALIZE OUTPUTS OF THE MODEL ###
######################################

##### Create histograms of metric values for different parameters per zone per timestep
zonesModelled = np.load(os.path.join(resultFolder, metricNames[0] + '.npy'))
histogramsModelledMetrics(metricNames)

print('1. Histograms for each zone and ech time step plotted.')  

##### Create colormap with absolute difference values

# Create an array corresponding to the zones
plotAbsoluteDifference(metricNames,'2000')
print('2. Difference betweeen the observed and modelled for each zone and ONE time step plotted.') 




  
  


