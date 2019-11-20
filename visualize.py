# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import calibrate

# Get metrics
metricNames = parameters.getSumStats()

# Get number of zones and parameters
numberOfZones = parameters.getNumberOfZones()
numberOfParameters = parameters.getNumberofIterations(
  parameters.getSuitFactorDict(), parameters.getParametersforCalibration()) # use this one only for colors

# Create colors for zones and parameters
zoneColors = plt.cm.rainbow(np.linspace(0,1,numberOfZones))
parameterColors = plt.cm.rainbow(np.linspace(0,1,numberOfParameters))

# Get the number of parameter iterations and number of time step defined in the parameter.py script
nrOfTimesteps = parameters.getNrTimesteps()
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsTimeSteps = parameters.getObsTimesteps()
observedYears = [parameters.getObsYears()[i] for i in obsTimeSteps]

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

tickDict = {
  'fontsize': VERY_SMALL,
  'fontweight': 'bold'}


#################
### FUNCTIONS ###
#################

def histogramsModelledMetrics(aVariable, zonesModelled):
  #('number of time steps: ',len(zonesModelled))
  #('number of parameter configurations: ',len(zonesModelled[0]))                     
  #('number of zones: ',len(zonesModelled[0][0][1]))
  
  for timeStep in timeSteps: # Loop data for each time step
    # Prepare the main plot for each timeStep
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey = True) # Depending on number of zones!
    hTitle = "Histogram of modelled metric "+aVariable+" for each zone in timestep: " + str(timeStep)
    fig.suptitle(hTitle, fontweight='bold')
    fig.subplots_adjust(hspace=1)
    
    for zone in range(numberOfZones): # Loop array to extraxt data for each zone
      # Prepare data for each zone to be plotted in each subplot
      metrics_for_histogram = []

      for i in range(0,len(zonesModelled[timeStep-1])): # Loop array to get metrics for each parameter configuration
        metrics_for_histogram.append(zonesModelled[timeStep-1][i][1][zone][0]) # [0] gives the raw number
        
      # Plot subplots
      axes.flatten()[zone].hist(metrics_for_histogram, bins = 'auto', )
      axes.flatten()[zone].set(title=zone+1)
      axes.flatten()[zone].ticklabel_format(axis='x', style='sci', scilimits=(-4,4))

    # Set the name and clear the directory if needed
    hName = 'Histogram_' + aVariable + "_modelled_timestep_"+ str(timeStep) + ".png"
    if os.path.exists(hName):
        os.remove(hName)

    # Save plot and clear    
    plt.savefig(os.path.join(resultFolder,hName))
    plt.clf()

    # Close all open figures
    plt.close('all')

''' Maybe for later
def transformArray(theArray):
  # Create an array storing absolute difference values in the shape of the zones
  rowNo = len(obsTimeSteps) # Number of rows is equal to number of observed years
  numberZones = len(theArray[0][0])
  parameterSets = len(theArray[0])
  # Create array
  n = int(np.sqrt(numberZones))
  newArray = np.zeros((rowNo,parameterSets,n,n))
  # Fill with values
  for row in range(0,rowNo):
    for pSet in range(0,parameterSets):
      for zone in range(0,numberZones):
        i = np.floor_divide(zone,n)
        j = np.remainder(zone,n)
        newArray[row][pSet][i][j] = theArray[row][pSet][zone] 
  return newArray

def plotAbsoluteDifference(metricNames, obsTimeStep):
  for aMetric in metricNames:
    differenceArray = np.load(os.path.join(resultFolder, aMetric + '_diff.npy'))
    array = transformArray(differenceArray)[1] # here [1] for year '2000' to be changed
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
    plt.close('all')'''
    
def getRMSEArray(m):
  rmseArray = np.load(os.path.join(resultFolder, m + '_RMSE.npy'))
  return rmseArray

def plotRMSEwithMeanRMSE_allMetrics():
  return None

def plotRMSEwithMeanRMSE(rArray, oneMetric, observations, parameterSets):
  # Values of metric np and pd give highly different results, so plot them using brakes in y axis:
  if oneMetric in ['np', 'pd']:
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.2,3.2))
  # Other metrics plot normally:
  else:
    f, (ax) = plt.subplots(1, 1, sharex=True, figsize=(3.2,3.2))

  f.suptitle('RMSE for modelled metric '+ oneMetric, y=0.95, fontweight='bold')
  plt.xticks(observations)
  plt.xlabel('observed years')
    
  meanIndex = calibrate.smallestMeanErrorIndex(rArray)
  for pSet in range(rArray.shape[1]):
    if pSet == meanIndex:
      myLabel = 'parameter set ' + str(pSet) + ': ' + str(parameterSets[pSet])
      lw = 2.0
      m = 2.5
    else:
      myLabel = None;
      lw = 0.3
      m = 1

    ax.plot(observations,rArray[:,pSet],'--o',
              linewidth = lw,
              label = myLabel,
              markersize=m,
              c = parameterColors[pSet],
              markerfacecolor=parameterColors[pSet]) 
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))

  # plot the same data on the second ax
    if oneMetric in ['np', 'pd']:
      ax2.plot(observations,rArray[:,pSet],
               '--o',
               linewidth = lw,
               label = myLabel,
               markersize=m,
               c = parameterColors[pSet])
      ax2.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
      # zoom-in / limit the view to different portions of the data
      limits = {
        'np':[[200,5000],[0,100]],
        'pd':[[20e-8, 300e-8],[0, 6e-8]]
        }
      ax.set_ylim(limits[oneMetric][0])  # outliers only
      ax2.set_ylim(limits[oneMetric][1])  # most of the data

      # hide the spines between ax and ax2
      ax.spines['bottom'].set_visible(False)
      ax2.spines['top'].set_visible(False)
      ax.xaxis.tick_top()
      ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top
      ax2.xaxis.tick_bottom()

      d = .015  # how big to make the diagonal lines in axes coordinates
      kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth = 1)
      ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
      ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

      kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
      ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
      ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

  # Create legend and y label
  if oneMetric in ['np', 'pd']:
    ax2.legend(
      title = 'The smallest mean RMSE:',
      bbox_to_anchor=(0, -0.65, 1, 0.2), # -0,65 is the offset along y axis
      loc="lower left",
      mode="expand",
      borderaxespad=0)._legend_box.align = "left"
    f.text(-0.01, 0.5, 'root mean square error', va='center', rotation='vertical')
    
  else:
    ax.legend(
      title = 'The smallest mean RMSE:',
      bbox_to_anchor=(0, -0.3, 1, 0.2),
      loc="lower left",
      mode="expand",
      borderaxespad=0)._legend_box.align = "left"
    plt.ylabel('root mean square error')

  # Set the name and clear the directory if needed
  rName = 'RMSE_' + oneMetric + '.png'
  rPath = os.path.join(resultFolder, rName)
  if os.path.exists(rPath):
      os.remove(rPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,rPath), bbox_inches = "tight",dpi=300)
  plt.close('all')
  
def plotBestparameterSets(arrayR, oneMetric,observations):
  # Find and plot parameter sets with the smallest RMSE
  fig = plt.figure(figsize=(3.2,3.2)) # Figure size set to give 8 cm of width
  plt.title('RMSE for the fitted parameter sets\nfor metric '+oneMetric,fontweight='bold')
  plt.ylabel('root mean square error')
      
  fitList = calibrate.smallestErrorInObsYear(arrayR)
  fitIndices = []
  for y in fitList:
    if y[1] not in fitIndices:
      fitIndices.append(y[1])
  labels = []
  theLabels =[]
  for i in range(len(fitIndices)):
    for y in fitList:
      if y[1] == fitIndices[i]:
        labels.append(parameters.getObsYears()[y[0]])
    theLabels.append(labels)
    labels=[]

  for year in range(len(fitIndices)):
    itsLabel = 'set %s: min RMSE in %s'%(fitIndices[year],theLabels[year])
    plt.plot(observations,arrayR[:,fitIndices[year]],'--o', label = itsLabel, linewidth = 1, markersize = 3)
    plt.xticks(observations)
    plt.xlabel('observed years')
    plt.legend()
  
  # Set the name and clear the directory if needed
  rName = 'RMSE_best_sets_' + oneMetric + '.png'
  rPath = os.path.join(resultFolder, rName)
  if os.path.exists(rPath):
      os.remove(rPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,rPath),bbox_inches = "tight",dpi=300)
  plt.close('all')


def plotObservedAndCalibratedMetrics(aMetric, observedYearArray, zonesObserved, zonesModelled, indexCalibratedSet):
  # Plot the observed values of metrics
 
  fig = plt.figure(figsize=(7.14,4)) # Figure size set to give 16 cm of width
  fig.suptitle('Observed and modelled values of metric '+ aMetric,fontweight='bold')
  
  ax1 = plt.subplot(121)
  ax1.set_title('observed')
  ax1.set_xticks(observedYearArray)
  #ax1.set_xticklabels(ax1.get_xticks(), tickDict)
  ax1.set_xlabel('observed years')
  ax1.set_ylabel('metric value')
  #ax1.set_yticklabels(ax1.get_yticks(), tickDict)

  ax2 = plt.subplot(122, sharey = ax1)
  ax2.set_title('modelled')
  xYears = np.arange(1990,2018+1, step=4)
  ax2.set_xticks(xYears)
  #ax2.set_xticklabels(xYears,fontdict = tickDict)
  ax2.set_xlabel('modelled years')
  #ax2.set_yticklabels(ax2.get_yticks(), tickDict)
 
  # Loop through the zones:
  for z in range(numberOfZones):
    # Plot observed values. Create and array and fill with values for each observed time step:
    metricValues = []
    for year in range(len(obsTimeSteps)):
      metricValues.append(zonesObserved[year][0][1][z][0])
    ax1.plot(observedYearArray,metricValues,'-o',linewidth = 0.7,
            markersize=1, label = 'Zone '+str(z),c=zoneColors[z])
    # Plot modelled values. Create and array and fill with values for each time step:
    modelledValues = []
    for year in range(len(zonesModelled[:,0])):    
      modelledValues.append(zonesModelled[year,indexCalibratedSet,1][z][0]) 
    ax2.plot(np.arange(1990,2018+1),modelledValues,'-o',linewidth = 0.7,
            markersize=1, label = 'Zone '+str(z),c=zoneColors[z])

  # place a text box in upper left in axes coords
  modText = 'parameter set ' + str(indexCalibratedSet) + ': ' + str(parameterSets[indexCalibratedSet])
  modProps = dict(boxstyle='round', facecolor='white', alpha=0.15)
  ax2.text(0.05, 0.95, modText, transform=ax2.transAxes,verticalalignment='top', bbox=modProps, fontsize = VERY_SMALL)

  # Move the legend outside of the plot box. Put it below, with the zones in two rows.
  plt.legend(
    bbox_to_anchor=(0.115,-0.22, 0.8, 0.2),
    loc="upper left",
    mode = "expand",
    bbox_transform=fig.transFigure,
    ncol=8)
  
  # Set the name and clear the directory if needed
  oName = aMetric + '_calc_and_obs.png'
  oPath = os.path.join(resultFolder, oName)
  if os.path.exists(oPath):
      os.remove(oPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,oPath), bbox_inches = "tight",dpi=300)
  plt.close('all')

def plotParameterValules():
  #print(plt.style.available)
  #plt.style.use('bmh')
  fig = plt.figure(figsize=(4,3.2))
  # Create list to store the lists of parameters
  p1=[]
  p2=[]
  p3=[]
  p4=[]

  for m in metricNames:
    zonesModelled = calibrate.getModelledArray(m)
    rmseArray = getRMSEArray(m)
    parameterSets = calibrate.getParameterConfigurations(zonesModelled) # gives the parameter sets
    calibratedIndex = calibrate.smallestMeanErrorIndex_2000_2006(rmseArray) # gives the index of the best parameter set
    print(parameterSets[calibratedIndex])
    p1.append(parameterSets[calibratedIndex][0])
    p2.append(parameterSets[calibratedIndex][1])
    p3.append(parameterSets[calibratedIndex][2])
    p4.append(parameterSets[calibratedIndex][3])

  # Number of metrics:
  N = len(metricNames)
  
  ind = np.arange(N)    # the x locations for the groups
  #width = 0.35       # the width of the bars: can also be len(x) sequence
  
  weight1 = plt.bar(ind, p1, label='NeighborSuitability', alpha=0.5)#, color=pColors[0])
  weight2 = plt.bar(ind, p2, bottom=p1,label='DistanceSuitability', alpha=0.5)#,color=pColors[1])
  weight3 = plt.bar(ind, p3, bottom=[0.75,0.75,0.75,0.25,0.0],label='TravelTimeCityBorder', alpha=0.5)#,color=pColors[2])
  weight4 = plt.bar(ind, p4, bottom=[0.75,0.75,0.75,0.50,0.5],label='CurrentLandUseSuitability', alpha=0.5)#,color=pColors[3])

  plt.title("Calibrated parameters", fontweight='bold')
  plt.ylabel('Parameters')
  plt.xlabel('Metrics')
  plt.xticks(ind, metricNames)
  plt.yticks(np.arange(0, 1.4, 0.25),[0,0.25,0.5,0.75,1])
  plt.legend(ncol=2)
  

  # Set the name and clear the directory if needed
  wName = 'metric_weights.png'
  wPath = os.path.join(resultFolder, wName)
  if os.path.exists(wPath):
      os.remove(wPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,wPath), bbox_inches = "tight",dpi=300)
  plt.close('all')
  

######################################
### VISUALIZE OUTPUTS OF THE MODEL ###
######################################
'''for theMetric in metricNames:
  zonesModelled = calibrate.getModelledArray(theMetric)
  zonesObserved = calibrate.getObservedArray(theMetric)
  rmseArray = getRMSEArray(theMetric)
  parameterSets = calibrate.getParameterConfigurations(zonesModelled) # gives the parameter sets
  calibratedIndex = calibrate.smallestMeanErrorIndex_2000_2006(rmseArray) # gives the index of the best parameter set
  
  ##### Create histograms of metric values for different parameters per zone per timestep
  
  #histogramsModelledMetrics(theMetric, zonesModelled)
  print('1. Histograms for each zone and ech time step plotted.')  

  """ Leave this for later use
  ##### Create colormap with absolute difference values
  # Create an array corresponding to the zones
  plotAbsoluteDifference(metricNames,'2000')
  print('2. Difference betweeen the observed and modelled for each zone and ONE time step plotted.')
  """
  ##### Plot RMSE
  plotRMSEwithMeanRMSE(rmseArray, theMetric, observedYears, parameterSets)
  
  plotBestparameterSets(rmseArray,theMetric,observedYears)
  print('2. RMSE plotted.')
  

  ##### Plot values of modelled and observed metrics
  plotObservedAndCalibratedMetrics(theMetric, observedYears, zonesObserved, zonesModelled, calibratedIndex)
  print('3. Observed and fitted metrics ploted.')'''

plotRMSEwithMeanRMSE_allMetrics()
plotParameterValules()

  
  
  







  
  


