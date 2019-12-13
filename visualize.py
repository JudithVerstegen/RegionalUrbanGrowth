# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import numpy as np
import parameters
from pcraster.framework import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import colors
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

tickDict = {
  'fontsize': VERY_SMALL,
  'fontweight': 'bold'}


#################
### FUNCTIONS ###
#################

def setNameClearSave(figName):
  # Set the name and clear the directory if needed
  wPath = os.path.join(resultFolder, 'Calibration_1_' + figName)
  if os.path.exists(wPath):
      os.remove(wPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,wPath), bbox_inches = "tight",dpi=300)
  plt.close('all')
  

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
    setNameClearSave('Histogram_' + aVariable + "_modelled_timestep_"+ str(timeStep) + ".png")
    
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
    setNameClearSave('Difference_' + aMetric + "_timestep_"+ str(obsTimeStep) + ".png")
    '''
    
def plotRMSE_calibrationPeriod(rArray, oneMetric, observations, parameterSets):
  # Values of metric np and pd give highly different results, so plot them using brakes in y axis:
  if oneMetric in ['np', 'pd']:
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.2,3.2))
  # Other metrics plot normally:
  else:
    f, (ax) = plt.subplots(1, 1, sharex=True, figsize=(3.2,3.2))

  f.suptitle('RMSE for modelled metric '+ oneMetric, y=0.95, fontweight='bold')
  plt.xticks(observations)
  plt.xlabel('observed years')
    
  meanIndex = calibrate.smallestMeanErrorIndex_2000_2006(rArray)
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
      title = 'The smallest mean RMSE for [2000, 2006]:',
      bbox_to_anchor=(0, -0.65, 1, 0.2), # -0,65 is the offset along y axis
      loc="lower left",
      mode="expand",
      borderaxespad=0)._legend_box.align = "left"
    f.text(-0.01, 0.5, 'root mean square error', va='center', rotation='vertical')
    
  else:
    ax.legend(
      title = 'The smallest mean RMSE for [2000, 2006]:',
      bbox_to_anchor=(0, -0.3, 1, 0.2),
      loc="lower left",
      mode="expand",
      borderaxespad=0)._legend_box.align = "left"
    plt.ylabel('root mean square error')

  # Set the name and clear the directory if needed
  setNameClearSave('RMSE_calibration_period_' + oneMetric + '.png')
  
def plotRMSE_eachTimeStep(arrayR, oneMetric,observations):
  # Find and plot parameter sets with the smallest RMSE
  fig = plt.figure(figsize=(3.2,3.2)) # Figure size set to give 8 cm of width
  plt.title('Smallest RMSE for the parameter sets\nfor metric '+oneMetric,fontweight='bold')
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
  setNameClearSave('RMSE_each_time_step_' + oneMetric + '.png')


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
  zonesNumber = len(zonesObserved[0][0][1])
  
  for z in range(zonesNumber):
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
    ax2.plot(np.arange(observedYears[0],observedYears[-1:][0]+1),modelledValues,'-o',linewidth = 0.7,
            markersize=1, label = 'Zone '+str(z),c=zoneColors[z])

  # place a text box in upper left in axes coords
  modText = 'parameter set ' + str(indexCalibratedSet) + ': ' + str(parameterSets[indexCalibratedSet])
  modProps = dict(boxstyle='round', facecolor='white', alpha=0.15)
  ax2.text(0.05, 0.95, modText, transform=ax2.transAxes,verticalalignment='top', bbox=modProps, fontsize = VERY_SMALL)

  # Move the legend outside of the plot box. Put it below, with the zones in two rows.
  if aMetric not in ['cilp','pd']:
    plt.legend(
      bbox_to_anchor=(0.115,-0.22, 0.8, 0.2),
      loc="upper left",
      mode = "expand",
      bbox_transform=fig.transFigure,
      ncol=8)
  
  # Set the name and clear the directory if needed
  setNameClearSave('obs_and_mod_'+aMetric+'.png')

def plotKappa_eachTimeStep():
  # Plot the kappa statistic for the best parameter sets for each year

  # Load data
  kappaArray = np.load(os.path.join(resultFolder, 'kappa.npy'))
  # Create the figure
  fig = plt.figure(figsize=(3.2,3.2)) # Figure size set to give 8 cm of width
  plt.title('Kappa coefficient of agreement',fontweight='bold')
  plt.ylabel('Kappa')
  # Create a list for      
  fitList = calibrate.biggestKappaInObsYear(kappaArray)
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
    itsLabel = 'set %s: max Kappa in %s'%(fitIndices[year],theLabels[year])
    plt.plot(observedYears,kappaArray[:,fitIndices[year]],'--o', label = itsLabel, linewidth = 1, markersize = 3)
    plt.xticks(observedYears)
    plt.xlabel('observed years')
    plt.legend()
   
  # Set the name and clear the directory if needed
  setNameClearSave('KAPPA_each_time_step.png')

def plotKappa_calibrationPeriod():
  # Plot the kappa statistic for all parameter sets

  # Load data
  kappaArray = np.load(os.path.join(resultFolder, 'kappa.npy'))
  kappaIndex = calibrate.biggestKappa_2000_2006(kappaArray)
  
  # Create the figure
  fig = plt.figure(figsize=(3.2,3.2)) # Figure size set to give 8 cm of width
  plt.title('Kappa coefficient of agreement',fontweight='bold')
  plt.ylabel('Kappa')       
  plt.xticks(observedYears)
  plt.xlabel('observed years')

  for p in range(len(kappaArray[0,:])):
    if p == kappaIndex:
      myLabel = 'parameter set ' + str(p) + ': ' + str("...")
      lw = 2.0
      m = 2.5
    else:
      myLabel = None;
      lw = 0.3
      m = 1

    plt.plot(observedYears,kappaArray[:,p],'--o',
             linewidth = lw,
             label = myLabel,
             markersize=m,
             c = parameterColors[p],
             markerfacecolor=parameterColors[p])          
    
  plt.legend()
  
  # Set the name and clear the directory if needed
  setNameClearSave('KAPPA_calibration_period.png')
  
def plotParameterValues(kapppaInx):
  # Number of metrics:
  nParams = len(parameters.getSuitFactorDict()[1])
  N = len(metricNames)+1 # = plus Kappa!

  
  # Create figure
  fig = plt.figure(figsize=(4,3.2))
  # Create list to store the lists of parameters
  weights = [[] for i in range(nParams)]

  for m in metricNames: 
    zonesModelled = calibrate.getModelledArray(m)
    zonesObserved = calibrate.getObservedArray(m)
    rmseArray = calibrate.calcRMSE(calibrate.createDiffArray(zonesModelled,zonesObserved), m)
    parameterSets = np.array(calibrate.getParameterConfigurations(zonesModelled)) # gives the parameter sets
    calibratedIndex = calibrate.smallestMeanErrorIndex_2000_2006(rmseArray) # gives the index of the best parameter set
    for j in range(nParams):
      weights[j].append(parameterSets[calibratedIndex][j])
  # Add Kappa
  for j in range(nParams):
    weights[j].append(parameterSets[kapppaInx][j])


  ind = np.arange(N)    # the x locations for the groups
  alpha = 0.5
  bottom = 0
  c = ['green','red','purple','orange']
  labels = ['NeighborSuitability', 'DistanceSuitability', 'TravelTimeCityBorder', 'CurrentLandUseSuitability']
  for bar in range(len(weights)):
    plt.bar(ind, weights[bar],label=labels[bar], bottom = bottom, alpha=alpha, color=c[bar])
    bottom = bottom + np.array(weights[bar])

  plt.title("Calibrated parameters", fontweight='bold')
  plt.ylabel('parameters')
  plt.xlabel('metrics')
  plt.xticks(ind, metricNames + ['kappa'])
  plt.yticks(np.arange(0, 1.5, 0.25),np.arange(0,1.1,0.25))
  plt.legend(ncol=2)

  setNameClearSave('parameters.png')

def plotUrbObs():
  # Plot the observed urban arreas
  ncol = 3
  nrows = int(np.ceil(len(observedYears)/ncol))
  # Load observed urb
  urb_obs = np.load(os.path.join(resultFolder, 'urb_obs.npy'))
  # Create figure
  fig = plt.figure(figsize=(7.14,2.5*nrows)) # Figure size set to give 16 cm of width
  fig.suptitle('Observed urban areas',fontweight='bold', y=0.95)
  # Create subplots

  for index, oYear in enumerate(observedYears):
    ax1 = plt.subplot(nrows,ncol,index+1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(oYear)
    urbMatrix1 = np.reshape(urb_obs[index,0,1], (1600,1600))
    ax1.xticks = ([])
    cmap = colors.ListedColormap(['black','yellow'])
    ax1.imshow(urbMatrix1)#, cmap=cmap)
  
  setNameClearSave('urb_observed.png')

def plotUrbMod(m, theIndex, parameterSets):
  # Plot the modelled urban arreas
  ncol = 3
  nrows = int(np.ceil(len(observedYears)/ncol))
  # Load modelled urb
  urb_mod = np.load(os.path.join(resultFolder, 'urb_subset.npy'))
  # Create figure
  fig = plt.figure(figsize=(7.14,2.5*nrows)) # Figure size set to give 16 cm of width
  fig.suptitle(
    'Modelled urban areas for parameter set '+str(theIndex)+': '+str(parameterSets[theIndex])
    ,fontweight='bold', y=0.95)
  
  # Create subplots
  for index, oYear in enumerate(observedYears):
    ax1 = plt.subplot(nrows,ncol,index+1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(oYear)
    urbMatrix1 = np.reshape(urb_mod[index,theIndex,1], (1600,1600))
    ax1.xticks = ([])
    cmap = colors.ListedColormap([(0.98,0.98,0.95),'black'])
    ax1.imshow(urbMatrix1, cmap=cmap)
  
  setNameClearSave('urb_modelled_'+str(theIndex)+'_'+m+'.png')

def plotUrbChanges():
  from matplotlib import colors
  # Plot the observed urban arreas
  ncol = 2
  nrows = 2
  # Load observed urb
  urb_obs = np.load(os.path.join(resultFolder, 'urb_obs.npy'))
  # Create figure
  fig = plt.figure(figsize=(7.14,7.14)) # Figure size set to give 16 cm of width
  fig.suptitle('Change in observed urban areas',fontweight='bold', y=0.95)
  # Create subplots

  for index, oYear in enumerate(observedYears[:-1]):
    ax1 = plt.subplot(nrows,ncol,index+1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(str(oYear) + '-' + str(observedYears[index+1]))
    changeMatrix = urb_obs[index+1,0,1] - urb_obs[index,0,1] 
    changeMatrix_reshape = np.reshape(changeMatrix, (1600,1600))
    ax1.xticks = ([])
    # Set colors for 'urban -> non-urban', 'no change', 'non-urban -> urban'
    cmap = colors.ListedColormap(['magenta', 'black','yellow'])
    bounds=[-1.5,-0.5,0.5,1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax1.imshow(changeMatrix_reshape, cmap=cmap)

  values = [-1,0,1]#np.unique(changeMatrix_reshape.ravel())
  labels = ['urban -> non-urban', 'no change', 'non-urban -> urban']#, 'missing value']
  colors = [ im.cmap(im.norm(value)) for value in values]
  patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
  # put those patched as legend-handles into the legend
  fig.legend(handles=patches,
             bbox_to_anchor=(0.069,0.03,0.78,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

  setNameClearSave('urb_changes_observed.png')

def plotUrbModelledChanges(m, calibratedIndex, parameterSets):
  # Plot the observed urban arreas
  ncol,nrows = 2,2

  # Load observed urb
  urb_mod_subset = np.load(os.path.join(resultFolder, 'urb_subset.npy'))
  # Create figure
  fig = plt.figure(figsize=(7.14,7.14)) # Figure size set to give 16 cm of width
  fig.suptitle('Change in modelled urban areas for parameter set '
               +str(calibratedIndex)+': '+str(parameterSets[calibratedIndex])+'\nmetric '+m,
               fontweight='bold', y=0.95)

  # Set colors
  colorsModelledChange = {
    -1: 'red',            # 'urban -> non-urban'
    0: (0.98,0.98,0.95),  # 'no change'
    1: 'black'}           # 'non-urban -> urban'


  labelsChange = {
    -1: 'urban -> non-urban',
    0: 'no change',
    1: 'non-urban -> urban'}
  
  # Create subplots
  states = []
  for obsIndex, theObsTimeStep in enumerate(obsTimeSteps[:-1]):
    ax1 = plt.subplot(nrows,ncol,obsIndex+1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(str(observedYears[obsIndex]) + '-' + str(observedYears[obsIndex+1]))
    ax1.xticks = ([])
    
    changeModMatrix = (urb_mod_subset[obsIndex+1,calibratedIndex,1]
                    - urb_mod_subset[obsIndex,calibratedIndex,1])  
    changeModMatrix_reshape = np.reshape(changeModMatrix, (1600,1600))
    u = np.unique(changeModMatrix[~numpy.isnan(changeModMatrix)])
    states.append(u)
    cmapL = [colorsModelledChange[v] for v in u]
    cmap = colors.ListedColormap(cmapL) 
    im = ax1.imshow(changeModMatrix_reshape, cmap=cmap)
  
  # Create legend

  values = np.unique(np.array([item for sublist in states for item in sublist]))
  cmap = colors.ListedColormap([colorsModelledChange[v] for v in values])
  labels = [labelsChange[v] for v in values]
  patchColors = [ colorsModelledChange[v] for v in values]
  patches = [ mpatches.Patch(color=patchColors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
  
  # put those patched as legend-handles into the legend
  fig.legend(handles=patches,
            bbox_to_anchor=(0.069,0.03,0.78,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
  
  setNameClearSave('urb_changes_modelled_'+str(calibratedIndex)+'_'+m+'.png') 
  



# This goes to visualize.py :
def plotTransition():
  # to be finished:
  '''transArray = state1+state2+state3+state4
  new = np.reshape(transArray,(1600,1600))

  plt.figure(figsize=(8,4))
  im = plt.imshow(new, interpolation='none')

  values = np.unique(new.ravel())
  labels = ['1. urban -> urban', '2. urban -> non-urban', '3. non-urban -> urban', '4. non-urban -> non-urban]']
  colors = [ im.cmap(im.norm(value)) for value in values]
  patches = [ mpatches.Patch(color=colors[i], label="State {l}".format(l=labels[i]) ) for i in range(len(values)) ]
  # put those patched as legend-handles into the legend
  plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
  plt.show()'''
  
######################################
### VISUALIZE OUTPUTS OF THE MODEL ###
######################################

###
print('1. Visualize the outputs of calculating the metrics')

### Print parameter values
kappaRun = False
kappaArray = np.load(os.path.join(resultFolder, 'kappa.npy'))
kappaIndex = calibrate.biggestKappa_2000_2006(kappaArray)
plotParameterValues(kappaIndex)

#### Print results for each metric
for theMetric in metricNames:
  zonesModelled = calibrate.getModelledArray(theMetric)
  zonesObserved = calibrate.getObservedArray(theMetric)
  rmseArray = calibrate.calcRMSE(calibrate.createDiffArray(zonesModelled,zonesObserved), theMetric)
  parameterSets = calibrate.getParameterConfigurations(zonesModelled) # gives the parameter sets
  calibratedIndex = calibrate.smallestMeanErrorIndex_2000_2006(rmseArray) # gives the index of the best parameter set
    
  ##### Create histograms of metric values for different parameters per zone per timestep
  #histogramsModelledMetrics(theMetric, zonesModelled) 

  ##### Plot RMSE
  plotRMSE_calibrationPeriod(rmseArray, theMetric, observedYears, parameterSets)
  plotRMSE_eachTimeStep(rmseArray,theMetric,observedYears)

  ##### Plot values of modelled and observed metrics
  plotObservedAndCalibratedMetrics(theMetric, observedYears, zonesObserved, zonesModelled, calibratedIndex)

  ##### Plot modelled urban areas and change in modelled urban areas
  plotUrbMod(theMetric, calibratedIndex, parameterSets)
  plotUrbModelledChanges(theMetric, calibratedIndex, parameterSets)
  
  ##### Plot kapppa results, but only once:
  if kappaRun == False:
    plotKappa_calibrationPeriod()
    plotKappa_eachTimeStep()
    plotUrbMod('kappa', kappaIndex, parameterSets)
    plotUrbModelledChanges('kappa', kappaIndex, parameterSets)
    
    kappaRun = True

###
print('2. Visualize the outputs of modelling the urban areas')

### Print observed values
plotUrbObs()
plotUrbChanges()


  
  
  
  
  







  
  


