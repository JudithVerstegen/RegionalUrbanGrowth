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
numberOfParameters = parameters.getNumberofIterations() # use this one only for colors

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

def setNameClearSave(scenario, figName):
  # Set the name and clear the directory if needed
  wPath = os.path.join(resultFolder, 'Scenario_'+str(scenario)+'_'+ figName)
  if os.path.exists(wPath):
      os.remove(wPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,wPath), bbox_inches = "tight",dpi=300)
  plt.close('all')
    
def plotRMSE(oneMetric, scenario):#(rArray, oneMetric, observations, parameterSets): #plotRMSE_calibrationPeriod
  # Plot the RMSE values for calibrated parameters for a gicen metric and scenario
  
  ## 1. Prepare the plot
  # Values of metric np and pd give highly different results, so plot them using brakes in y axis:
  if oneMetric in ['np', 'pd']:
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.2,3.2))
  # Other metrics plot normally:
  else:
    f, (ax) = plt.subplots(1, 1, sharex=True, figsize=(3.2,3.2))

  f.suptitle('RMSE for modelled metric '+ oneMetric, y=0.95, fontweight='bold')
  plt.xticks(observedYears)
  plt.xlabel('observed years')
  ## 2. Get the data
  # Get the RMSE values for validation
  rArray = calibrate.calcRMSE(oneMetric, scenario, 'validation')
  # Get the parameter sets
  parameterSets = calibrate.getParameterConfigurations()
  # Get the calibrated index
  meanIndex = calibrate.getCalibratedIndeks(oneMetric,scenario)
  ## 3. Plot the data
  # Loop parameter sets
  for pSet in range(rArray.shape[1]):
    # Underline the RMSE for the calibrated index
    if pSet == meanIndex:
      myLabel = 'parameter set ' + str(pSet) + ': ' + str(parameterSets[pSet])
      lw = 2.0
      m = 2.5
    # For other parameters, plot smaller lines
    else:
      myLabel = None;
      lw = 0.3
      m = 1
    # For each parameter set plot RMSE
    ax.plot(observedYears,rArray[:,pSet],'--o',
              linewidth = lw,
              label = myLabel,
              markersize=m,
              c = parameterColors[pSet],
              markerfacecolor=parameterColors[pSet]) 
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))

  # plot the same data on the second ax
    if oneMetric in ['np', 'pd']:
      ax2.plot(observedYears,rArray[:,pSet],
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

  ## 4. Create legend and y label
  # Get the calibration period
  period = parameters.getCalibrationPeriod()[scenario]['calibration']
  if oneMetric in ['np', 'pd']:
    ax2.legend(
      title = 'The smallest mean RMSE for '+str(period)+':',
      bbox_to_anchor=(0, -0.65, 1, 0.2), # -0,65 is the offset along y axis
      loc="lower left",
      mode="expand",
      borderaxespad=0)._legend_box.align = "left"
    f.text(-0.01, 0.5, 'root mean square error', va='center', rotation='vertical')
    
  else:
    ax.legend(
      title = 'The smallest mean RMSE for '+str(period)+':',
      bbox_to_anchor=(0, -0.3, 1, 0.2),
      loc="lower left",
      mode="expand",
      borderaxespad=0)._legend_box.align = "left"
    plt.ylabel('root mean square error')

  # Set the name and clear the directory if needed
  setNameClearSave(scenario,'RMSE_' + oneMetric + '.png')
  
def plotObservedAndCalibratedMetrics(aMetric, scenario):#, observedYearArray, zonesObserved, zonesModelled, indexCalibratedSet):
  # Plot the observed values of metrics in one subplot
  # and the metric values for the calibrated parameters on the other subplot

  ## 1. Get the data
  zonesObserved = calibrate.getObservedArray(metric,scenario,aim='validation')
  zonesModelled = calibrate.getModelledArray(metric,scenario,aim='validation')
  indexCalibratedSet = calibrate.getCalibratedIndeks(metric,scenario)
  parameterSets = calibrate.getParameterConfigurations()

  ## 2. Prepare the plot  
  fig = plt.figure(figsize=(7.14,4)) # Figure size set to give 16 cm of width
  fig.suptitle('Observed and modelled values of metric '+ aMetric,fontweight='bold')
  
  ax1 = plt.subplot(121)
  ax1.set_title('observed')
  ax1.set_xticks(observedYears)
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

  ## 3. Plot the values
  # Loop through the zones:
  zonesNumber = len(zonesModelled[0][0][1])
  
  for z in range(zonesNumber):
    # Plot observed values. Create and array and fill with values for each observed time step:
    metricValues = []
    for year in range(len(obsTimeSteps)):
      metricValues.append(zonesObserved[year][0][1][z][0])
    ax1.plot(observedYears,metricValues,'-o',linewidth = 0.7,
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
  setNameClearSave(scenario, 'obs_and_mod_'+aMetric+'.png')

def plotKappa(scenario): #plotKappa_calibrationPeriod
  # Plot the kappa statistic for all parameter sets

  # Load data
  kappaArray = calibrate.getKappaArray(scenario,aim='validation')
  kappaIndex = calibrate.getCalibratedIndeks('kappa',scenario)
  parameterSets = getParameterConfigurations()
  
  # Create the figure
  fig = plt.figure(figsize=(3.2,3.2)) # Figure size set to give 8 cm of width
  plt.title('Kappa Standard '+str(scenario),fontweight='bold')
  plt.ylabel('Kappa')       
  plt.xticks(observedYears)
  plt.xlabel('observed years')
  # Loop parameter sets
  for p in range(len(kappaArray[0,:])):
    # underline calibrated set
    if p == kappaIndex:
      myLabel = 'parameter set ' + str(p) + ': ' + str(parameterSets[p])
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
  setNameClearSave(scenario,'kappa.png')
  
def plotParameterValues(scenario):
  # Number of metrics:
  nParams = len(parameters.getSuitFactorDict()[1])
  N = len(metricNames)+1 # = plus Kappa!
  # Get the all possible parameter sets
  parameterSets = calibrate.getParameterConfigurations()
  
  # Create figure
  fig = plt.figure(figsize=(4,3.2))
  # Create list to store the lists of parameters
  weights = [[] for i in range(nParams)]
  # Loop metrics
  for m in metricNames:
    # get the index of the best parameter set
    calibratedIndex = calibrate.getCalibratedIndeks(m,scenario)
    for j in range(nParams):
      weights[j].append(parameterSets[calibratedIndex][j])
  # Add Kappa
  kapppaInx = calibrate.getCalibratedIndeks('kappa',scenario)
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

  setNameClearSave(scenario,'parameters.png')

def plotUrbObs():
  # Plot the observed urban arreas
  ncol = 3
  nrows = int(np.ceil(len(observedYears)/ncol))
  # Load observed urb
  urb_obs = calibrate.getObservedArray('urb')
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
  
  setNameClearSave('observed','urb.png')

def plotUrbMod(metric,scenario):# theIndex, parameterSets):
  theIndex = calibrate.getCalibratedIndeks(metric,scenario)
  parameterSets = calibrate.getParameterConfigurations()
  # Plot the modelled urban arreas
  ncol = 3
  nrows = int(np.ceil(len(observedYears)/ncol))
  
  # Create figure
  fig = plt.figure(figsize=(7.14,2.5*nrows)) # Figure size set to give 16 cm of width
  fig.suptitle(
    'Modelled urban areas for parameter set '+str(theIndex)+': '+str(parameterSets[theIndex]),
    fontweight='bold', y=0.95)
  
  # Create subplots
  for index, oYear in enumerate(observedYears):
    # Load modelled urban areas. Each year is saved in the seperate npy file.
    urb_mod = calibrate.getModelledArray('urb_subset_'+str(obsTimeSteps[index]))
    ax1 = plt.subplot(nrows,ncol,index+1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(oYear)
    urbMatrix1 = np.reshape(urb_mod[0,theIndex,1], (1600,1600))
    ax1.xticks = ([])
    cmap = colors.ListedColormap([(0.98,0.98,0.95),'black'])
    ax1.imshow(urbMatrix1, cmap=cmap)
  
  setNameClearSave(scenario,'urb_modelled_'+str(theIndex)+'_'+metric+'.png')

def plotUrbChanges():
  from matplotlib import colors
  # Plot the observed urban arreas
  ncol = 2
  nrows = 2
  # Load observed urb
  urb_obs = calibrate.getObservedArray('urb')
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

  setNameClearSave('observed','urb_changes.png')

def plotUrbModelledChanges(metric,scenario):
  calibratedIndex = calibrate.getCalibratedIndeks(metric,scenario)
  parameterSets = calibrate.getParameterConfigurations()
  
  # Plot the observed urban arreas
  ncol,nrows = 2,2

  # Create figure
  fig = plt.figure(figsize=(7.14,7.14)) # Figure size set to give 16 cm of width
  fig.suptitle('Change in modelled urban areas for parameter set '
               +str(calibratedIndex)+': '+str(parameterSets[calibratedIndex])+'\nmetric '+metric,
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
    # Load observed urb
    urb_mod_current = calibrate.getModelledArray('urb_subset_'+str(theObsTimeStep))
    urb_mod_next = calibrate.getModelledArray('urb_subset_'+str(obsTimeSteps[obsIndex+1]))
    ax1 = plt.subplot(nrows,ncol,obsIndex+1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(str(observedYears[obsIndex]) + '-' + str(observedYears[obsIndex+1]))
    ax1.xticks = ([])
    
    changeModMatrix = (urb_mod_next[0,calibratedIndex,1]
                    - urb_mod_current[0,calibratedIndex,1])  
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
  
  setNameClearSave(scenario,'urb_changes_modelled_'+str(calibratedIndex)+'_'+metric+'.png') 
  
def plotBestPerformers(metric, scenario, nrOfBestPerformers):  
  nParams = len(parameters.getSuitFactorDict()[1])
  x=['suitfactor'+str(i) for i in range(1,nParams+1)]
  labels = [ 'rank '+str(i) for i in range(1,nrOfBestPerformers+1)]
  topperformers = calibrate.getTopCalibratedParameters(metric,scenario,nrOfBestPerformers)
  y = topperformers[:,3:].astype('float64')

  plt.title('metric '+metric + ' top 20', fontweight = 'bold')
  plt.ylabel('parameters')
  plt.xlabel('suitability factors')
  plt.xticks([1,2,3,4],['NeighborSuitability', 'DistanceSuitability', 'TravelTimeCityBorder', 'CurrentLandUseSuitability'])
  plt.boxplot(y,whis=[5,95]) # whis: set the whiskers at specific percentiles of the data [p1,p2] <- percentiles
  setNameClearSave(scenario, metric+'_top_performers.png') 


def plotMultiobjective(scenario):
  # Plot the results of multi-objective calibration.
  # Each plot will presente the results of all 3 case studies, for each calibration scenario separately.

  # Crete a dict to store the multiobjective results for each country
  multiDict = {}
  for country in ['IE','IT','PL']:
    fileName = 'scenario_'+str(scenario)+'_multiobjective.npy'
    multiobjectiveArray = os.path.join(os.getcwd(),'results',country, 'metrics',fileName)
    # The file contains validation results. Rows: index, RMSE, kappa, goal function
    multiDict[country] = np.load(multiobjectiveArray)
  # Create a dict to store the improvement of the multiobjective results compared to the cell-based goal function
  imprDict = {}
  for country in ['IE','IT','PL']:
    imprArray = multiDict[country]
    for metric in range(0,len(metricNames)):
    # Find the improvement for RMSE
      cell_based_RMSE = multiDict[country][metric,0][1,0]
      imprArray[metric,0][1,:] = (cell_based_RMSE-imprArray[metric,0][1,:])/cell_based_RMSE*100
      # Find the improvement for Kapppa
      cell_based_Kappa = multiDict[country][metric,0][2,0]
      imprArray[metric,0][2,:] = (imprArray[metric,0][2,:]-cell_based_Kappa)/cell_based_Kappa*100
      # Find the improvement for goal function (just in case)
      cell_based_gf = multiDict[country][metric,0][3,0]
      imprArray[metric,0][3,:] = (imprArray[metric,0][3,:]-cell_based_gf)/cell_based_gf*100
    imprDict[country] = imprArray
    
  plt.title('Scenario '+str(scenario)+' multiobjective goal function results for 3 case studies',fontweight='bold')
  plt.ylabel('Kappa Standard improvement [%]')
  plt.xlabel('RMSE improvement [%]')
  plt.axvline(x=0, alpha=0.2)
  plt.axhline(y=0, alpha=0.2)
  plotC = {
    1:'blue',
    2:'red',
    3:'green',
    4:'yellow'}
  for country in ['IE','IT','PL']:
    for m in range(0,len(metricNames)):
      if country == 'IE':
        plt.scatter(imprDict[country][m,0][1,:-1],multiDict[country][m,0][2,:-1],c=plotC[m+1],label=metricNames[m])
      else:
        plt.scatter(imprDict[country][m,0][1,:-1],multiDict[country][m,0][2,:-1],c=plotC[m+1])

  plt.legend()
  setNameClearSave(scenario, 'multiobjective') 

plotMultiobjective(1)
plotMultiobjective(2)

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
'''
for scenario in [1,2,3]:
  print('Scenario',scenario)
  print('1. Visualize the outputs of calculating the metrics')

  ### Print parameter values
  plotParameterValues(scenario)

  #### Print results for each metric
  for metric in metricNames:
    #plotRMSE(metric,scenario)
    #plotObservedAndCalibratedMetrics(metric, scenario)
    #plotUrbMod(metric,scenario)
    #plotUrbModelledChanges(metric, scenario)
    plotBestPerformers(metric, scenario, 20)

  ###
  print('2. Visualize the outputs of calculating Kapppa standard')
  #plotKappa(scenario)
  plotBestPerformers('kappa', scenario, 20)
  
  ###
  print('3. Visualize the outputs of modelling the urban areas')

### Print observed values
#plotUrbObs()
#plotUrbChanges()
'''
  
''' Maybe for later
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

def plotRMSE_eachTimeStep(arrayR, oneMetric,observations):
  # Find and plot parameter sets with the smallest RMSE
  fig = plt.figure(figsize=(3.2,3.2)) # Figure size set to give 8 cm of width
  plt.title('Smallest RMSE for the parameter sets\nfor metric '+oneMetric,fontweight='bold')
  plt.ylabel('root mean square error')
      
  fitList = calibrate.RMSEindex_eachTimeStep(arrayR)
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
  setNameClearSave('KAPPA_each_time_step.png')'''
  
  
  







  
  


