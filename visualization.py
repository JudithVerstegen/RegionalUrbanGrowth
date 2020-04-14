# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import metrics
import numpy as np
import parameters
import calibrate
from pcraster.framework import *
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

#### Script to read in the metrics saved as the result of the LU_urb.py script.
#### Metrics are transformed into an array

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
c_mod = ['lavender', 'indigo', 'gold']
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

#################
### FUNCTIONS ###
#################

def setNameClearSave(figName, scenario=None):
  # Set the name and clear the directory if needed
  if scenario is None:
    name = ''
  else:
    name = '_scenario_'+str(scenario)
  # Create figure dir
  fig_dir = os.path.join(os.getcwd(),'results','figures')
  if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
  # Create dir of a file
  wPath = os.path.join(fig_dir, figName+name)
  if os.path.exists(wPath):
      os.remove(wPath)

  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,wPath), bbox_inches = "tight",dpi=300)
  plt.close('all')

def getResultsArrayEverySet():
  results = np.empty((len(all_metrices),6,165)) # 6=3 case studies * 2 scenarios, 165 parameter sets
  for i,m in enumerate(all_metrices):
    j=0
    for country in case_studies:
      for scenario in [1,2]:
        # an_array is an array of arrays (contains rmse/kappa arrays for every p set)
        if m == 'kappa':
          an_array = calibrate.getKappaArray(scenario,aim,case=country)
        else:
          an_array = calibrate.calcRMSE(m,scenario,aim,case=country)
        results[i,j] = an_array
        j+=1
  return results

def getAverageResultsArrayEverySet(array,aim):
  results = np.empty((len(all_metrices),6,165))
  for i in all_metrices:
    j=0
    for country in case_studies:
      for scenario in [1,2]:
        # get the goal function value of the metric for every parameter set
        av_results[i,j] = calibrate.getAveragedArray(array[i,j],scenario,aim)
  return av_results

def getValidationResults():
  """
  Create an array matrix: validation metrics x goal functions (metrics determining parameter set) * 6
  """
  goal_functions = all_metrices
  validation_metrices = all_metrices + ['simulation','allocation']
  # validation on 3 case studies and 2 scenarios
  validationResults = np.empty((len(validation_metrices),len(goal_functions)*3*2))
  # Lop validation metrices, to get their values:
  for i,v_m in enumerate(validation_metrices):
    j=0
    for m in goal_functions:
      for country in case_studies:
        for scenario in [1,2]:
          # Get the index of the calibrated parameter set
          index = calibrate.getCalibratedIndeks(m, scenario,country)
          
          # an_array is an array of arrays (contains rmse/kappa arrays for every p set)
          if v_m == 'kappa':
            an_array = calibrate.getKappaArray(scenario,'validation',case=country)
          elif v_m == 'simulation':
            an_array = calibrate.getKappaSimulationArray('validation',case=country)
          elif v_m == 'allocation':
            an_array = calibrate.getAllocationArray('validation',case=country)
          else:
            # Get the error for the validation metric
            an_array = calibrate.calcRMSE(v_m,scenario,'validation',case=country)
          # Get the average values for the vaidation period
          av_results = calibrate.getAveragedArray(an_array,scenario,'validation')
          validationResults[i,j] = av_results[index]
          j+=1
     
  return validationResults  

def createNormalizedResultDict(metrics,aim):
  # Create array to store calibration/validation results for all metrics
  results = {}
  for c in case_studies:
    results[c]={}
    for s in [1,2]:
      results[c][s] = np.empty([165,len(metrics)])   
  
  # Loop all the countries:
  for country in case_studies:
    # Get data
    resultFolder = os.path.join(os.getcwd(),'results',country, 'metrics')
    parameterSets = calibrate.getParameterConfigurations()
    parameters=range(0,len(parameterSets))

    i=0
    # Loop all the metrics:
    for m in metrics:
      # Loop calibration scenarios:
      for scenario in [1,2]:
        # Load data
        if m == 'kappa':
          an_array = calibrate.getKappaArray(scenario,aim,case=country)
        else:
          an_array = calibrate.calcRMSE(m,scenario,aim,case=country)

        # get the calibration value of the metric for every set
        av_array = calibrate.getAveragedArray(an_array,scenario,aim)
        if m is 'kappa':
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

#############################
########### PLOTS ###########
#############################

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

def plotUrbChangesMod(): #DONE
  """
  Plot the modelled changes in urban areas in years 2000-2006 (timestep 11 and 17),
  for all metrics, 3 case studies but only one scenario (1).
  First column on the left presents the observed chages.
  18 subplots stacked in 3x6 matrix.
  """
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(3,6, sharex=True, sharey=True, figsize=(7.6,4)) # 16 cm of width
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  # Set colors
  cmap_obs = colors.ListedColormap(c_obs)
  colorsModelledChange = {-1: c_mod[0],0: c_mod[1],1: c_mod[2]}
  
  ## 2. Loop and plot modellled countries
  # Select the time steps
  selected_time_steps = [11,17] #(index 2 and 3)
  scenario=2
  i=0
  for country in case_studies:
    # And add the map for the observed change! Get the data:
    urb_obs = calibrate.getObservedArray('urb', case=country)
    changeMatrix = urb_obs[3,0,1] - urb_obs[2,0,1]
    # Reshape the 1D array into a map
    changeMatrix_reshape = np.reshape(changeMatrix, (1600,1600))
    # Plot the observed changes
    axs[i,0].imshow(changeMatrix_reshape, cmap=cmap_obs)
    axs[i,0].set(xlabel='observed', ylabel=country+str(scenario))
    # Loop metrics and for each metric get the change map for seected country, and selected metric (goal function)    
    j=1
    for metric in all_metrices:
      calibratedIndex = calibrate.getCalibratedIndeks(metric,scenario,case=country)
      # Get the data for year 2000 and 2006
      urb_mod_current = calibrate.getModelledArray('urb_subset_'+str(selected_time_steps[0]),case=country)
      urb_mod_next = calibrate.getModelledArray('urb_subset_'+str(selected_time_steps[1]),case=country)          
      changeModMatrix = (urb_mod_next[0,calibratedIndex,1]
                      - urb_mod_current[0,calibratedIndex,1])
      # Reshape the matrix into 2D
      changeModMatrix_reshape = np.reshape(changeModMatrix, (1600,1600))
      # Find states in the map to adjust the colors:
      u = np.unique(changeModMatrix[~numpy.isnan(changeModMatrix)])
      cmapL = [colorsModelledChange[v] for v in u]
      cmap = colors.ListedColormap(cmapL)
      # Plot
      axs[i,j].imshow(changeModMatrix_reshape, cmap=cmap)
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

def plotBestPerformers(): #Done
  """
  Plot five subplots, each for a goal function.
  Each subplot shows boxplots for 15% of best parameter sets, for 3 case studies and 2 scenarios
  """
  nrOfBestPerformers = 25 #~15% of all parameters
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(5, figsize=(8,8),sharex=True) # 16 cm of width
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
  
  for m_i,metric in enumerate(all_metrices):
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
        
        topperformers = calibrate.getTopCalibratedParameters(metric,scenario,nrOfBestPerformers,country)
        y = topperformers[:,3:].astype('float64')
        # Set subplot title
        if metric == 'kappa':
          axs[i].set_title(metric[0].upper(), pad=2)
        else:
          axs[i].set_title(metric.upper(), pad=2)
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
  
  setNameClearSave(str(nrOfBestPerformers)+'_top_performers')
plotBestPerformers()

def plotGoalFunctionEverySet(): #DONE
  """
  Plot the values for a goal function durnig calibration. Values for every parameter set tested
  5 subplots stacked vertically, one for every metric
  """

  # First, get all results
  r = getResultsArrayEverySet()
  results = getAverageResultsArrayEverySet(r,'calibration')
  # Loop the data for all metrics to get minimum and maximum goal function values
  limits = {}
  for i,m in enumerate(all_metrices):
    limits[m] = {
      'min': np.amin(results[i]),
      'max': np.amax(results[i])}

  # Now, plot the data  
  ## 1. Get the parameter sets
  parameterSets = calibrate.getParameterConfigurations()
  parameters=np.arange(0,len(parameterSets),1)
  n = len(metricNames)+1 # number of subplots
  
  ## 2. Prepare the plot  
  fig, axs = plt.subplots(n, figsize=(7.14,4), sharex = True) # 16 cm of width
  xticks = np.arange(0, parameters[-1]+10, 15.0)
  plt.xticks(xticks,[int(x) for x in xticks])
  plt.xlabel('parameter set')
  fig.align_ylabels()
  
  ## 3. Loop metrics. Each metric = new subplot
  for i,m in enumerate(metricNames+['kappa']):
    j=0
    ylable = m.upper()
    if m == 'kappa':
        ylable = m[0].upper()
    axs[i].set_ylabel(ylable)  
    # Loop all the countries. Each suplot has data for all case studies:
    for country in case_studies:
      # Loop calibration scenarios:
      for scenario in [1,2]:
        # set the min and max y axis values:
        amin = limits[m]['min']
        amax = limits[m]['max']
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
        axs[i].plot(parameters,results[i,j],
                 fmt[scenario],
                 linewidth = 0.5,
                 label = myLabel[scenario],
                 c = countryColors[country])
        j+=1

  # Create the legend
  leg = fig.legend(
    bbox_to_anchor=(0., 1.25, 1, .102),
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
  fig = plt.figure(figsize=(8,4))
  ax=[ fig.add_subplot(111) for i in [1,2] ] # Add subplot for each scenario
  drivers = ['NEIGH', 'TRAIN', 'TRAVEL', 'LU']
  ind = np.arange(len(all_metrices))    # the x locations for the groups
  width = 0.13 # width of a bar
  space = 0.02 # space between case studies
  alpha = {1:0.9,2:0.6}
  parameterSets = calibrate.getParameterConfigurations()
  #alist = 0
  #plt.title("Calibrated parameters", fontweight='bold')
  plt.ylabel('parameters')
  plt.ylim([0,1.1])
  plt.xlim([-0.5,5.25])
  fig.text(0.42, 0.03,
           "goal function")
           #ha='center')
  #plt.xlabel('goal function')
  plt.xticks(ind, ['f('+x.upper()+')' for x in metricNames] + ['h1(K)'])
  #plt.yticks(np.arange(0, 1.1, 0.25),np.arange(0,1.5,0.25))

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
  #leg11 = plt.legend([axs[1][0],axs[2][0]], ['scenario 1','scenario 2'], loc=1)
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

def plotObsAndMod():
  """
  # Plots two subplots for each metric
  # Plot the observed values of metrics in one subplot
  # and the modelled metric values for the calibrated parameters on the other subplot
  # All landscape metrics = 4 plots, 2 subplots each, stacked vertically
  # For FDI and WFDI plots values for zones on the diagonal of the case study area (zones: 0,5,10,15)
  """
  ## 1. Create the figure
  fig = plt.figure(figsize=(7.14,16)) # Figure size set to give 16 cm of width
  fig.align_ylabels()
  
  # Set linestyle for each scenario
  linestyle={1:'o-',2:'o--'}
  ## 2. Get results for countries, scenarios and all metrics (with kappa)
  parameterSets = calibrate.getParameterConfigurations()
  i=1
  for m,metric in enumerate(metricNames):
    axs1 = plt.subplot(4,2,i)
    axs2 = plt.subplot(4,2,i+1, sharey=axs1)
    plt.subplots_adjust(wspace=0.2, hspace=0.35)
    for country in case_studies:
      zonesObserved = calibrate.getObservedArray(metric,case=country)
      # Prepare the 'observed' and 'modelled' subplots
      
      axs1.set_title('Observed')
      axs2.set_title('Goal function '+metric.upper())
      
      axs1.set_xticks(observedYears)
      axs1.set_ylabel(metric.upper())
      if metric=='kappa':
        axs1.set_ylabel(metric[0].upper())
      axs1.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
      
      xYears = np.arange(1990,2018+1, step=4)
      axs2.set_xticks(xYears)
      axs2.set_ylabel(None)
      axs2.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        
      for scenario in [1,2]:
        zonesModelled = calibrate.getModelledArray(metric,scenario,case=country)
        indexCalibratedSet = calibrate.getCalibratedIndeks(metric,scenario,case=country)

        ## 3. Plot the values
        # Select the zones for plottong. Some metric are calculted for 16 zones. Other for one zone only:
        zonesNumber = len(zonesModelled[0][0][1])
        if zonesNumber > 1:
          selected_zones = [0,5,10,15]
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

'''def plotFindingWeights(): # works, but no point
  """
  Creates 4 subplots, gor each metric based goal function
  Each subplot presents the validation results for a multiobjective goal function combining metric based
  and locational based goal functions
  """

  ## 1, Crete a dict to store the multiobjective results for each country
  multiDict = {}
  for scenario in [1,2]:
    multiDict[scenario]={}
    for country in case_studies:
      fileName = 'scenario_'+str(scenario)+'_multiobjective.npy'
      multiobjectiveArray = os.path.join(os.getcwd(),'results',country, 'metrics',fileName)
      # The file contains validation results. Rows: index, RMSE, kappa, goal function
      multiDict[scenario][country] = np.load(multiobjectiveArray)

  ## 2. Create a dict to store the normalized values
  normDict = {}
  for scenario in [1,2]:
    normDict[scenario]={}
    for country in case_studies:
      normArray = multiDict[scenario][country]
      for metric in range(0,len(metricNames)):
        ## Find the normalized value of RMSE for each weight combination 
        normArray[metric,0][1,:] = calibrate.getNormalizedArray(normArray[metric,0][1,:],kappa=False)
        ## Find the normalized value of Kappa for each weight combination 
        normArray[metric,0][2,:] = calibrate.getNormalizedArray(normArray[metric,0][2,:],kappa=True)
      normDict[scenario][country] = normArray

  ## 3. Prepare the plot!
  fig, axs = plt.subplots(2,2, figsize=(7.14,4)) # 16 cm of width
  linestyle = {1:'solid',2:'dashed'}
  for scenario in [1]:
    i=0
    j=0
    
    for m,metric in enumerate(metricNames):
      for country in case_studies:
        axs[i,j].plot(
          normDict[scenario][country][m,0][1,1:-1],
          c = countryColors[country])
        axs[i,j].plot(
          normDict[scenario][country][m,0][2,:-1],
          '--o',
          c = countryColors[country])
      j=+1
      if m==1:
        i=1
        j=0
  plt.show()'''

def plotValidationResults(): 
  """
  Plot bars presenting validation results (y axis) for 5 goal funcstions (x axis)
  There are 5 goal functions, 3 case studies, 2 scenarios --> 15 bars
  There are 7 validation metrics --> 7 subplots
  """
  # Get the array with validation results:
  results = getValidationResults()

  # Create figure
  fig, axs = plt.subplots(7,1,figsize=(8,8),sharex=True) # 7 validation metrices
  validation_metrices = ['f('+x.upper()+')' for x in metricNames] + ['h1(K)','h2(Ks)','h3(A)']
  ind = np.arange(len(all_metrices))    # the x locations for the groups
  n = 3*2 # 3 countries, two scenarios
  width = 0.1 # width of a bar
  space = 0.02 # space between case studies
  alpha = {1:0.9,2:0.6}
  plt.xlabel('goal function')
  plt.xticks(ind, validation_metrices)
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
  ylabel=['RMSE','RMSE','RMSE','RMSE','K','Ks','A']
  # Prepare colors:
  bar_c=[]
  for case in case_studies:
    for s in [1,2]:
      a_color = colors.to_rgba(countryColors[case])[:-1]+(alpha[s],)
      bar_c.append(a_color)
  # Now plot the bars. For each metric, the bars are plotted as a different ax
  for v_m,v_metric in enumerate(validation_metrices):
    axs[v_m].set_title(validation_metrices[v_m], pad=2)
    axs[v_m].set_ylabel(ylabel[v_m])
    axs[v_m].bar(
        positions,
        results[v_m],
        color = bar_c, 
        width=width)
    # Draw lines dividing scenario bars and add annotation with the country symbol
    for xtick in ind:
      for c,country in enumerate(case_studies):
        p = xtick-2*width-space+c*(2*width+space)
        axs[v_m].axvline(p, alpha=0.5,c='white',linestyle='--', linewidth=0.5)
    
  '''#Create a legend
  handles, labels = axs[0].get_legend_handles_labels()
  patch = handles[0][0]
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
    lh.set_alpha(alpha[i+1])'''
  
  # Set the name and clear the directory if needed
  plt.show()#setNameClearSave('plotParameters', scenario=None)
plotValidationResults()








def plotImprovementToCellBased(base):
  """
  # Plot the validation results of multi-objective foal function used in calibration.
  # index defines, what is the base for comparison:
  'cell': 0 <- results for cell-based  goal function
  'patch': -1 <- results for patch based goal function
  """
  index = {'cell': 0,
           'patch': -1 }
  # Crete a dict to store the multiobjective results for each country
  multiDict = {}
  for scenario in [1,2]:
    multiDict[scenario]={}
    for country in case_studies:
      fileName = 'scenario_'+str(scenario)+'_multiobjective.npy'
      multiobjectiveArray = os.path.join(os.getcwd(),'results',country, 'metrics',fileName)
      # The file contains validation results. Rows: index, RMSE, kappa, goal function
      multiDict[scenario][country] = np.load(multiobjectiveArray)
  # Create a dict to store the improvement of the multiobjective results compared to the cell-based goal function
  imprDict = {}
  for scenario in [1,2]:
    imprDict[scenario] = {}
    for country in case_studies:
      imprArray = multiDict[scenario][country]
      for metric in range(0,len(metricNames)):
      # Find the improvement for RMSE
        cell_based_RMSE = multiDict[scenario][country][metric,0][1,index[base]]
        imprArray[metric,0][1,:] = (cell_based_RMSE-imprArray[metric,0][1,:])/cell_based_RMSE*100
        # Find the improvement for Kapppa
        cell_based_Kappa = multiDict[scenario][country][metric,0][2,index[base]]
        imprArray[metric,0][2,:] = (imprArray[metric,0][2,:]-cell_based_Kappa)/cell_based_Kappa*100
        # Find the improvement for goal function (just in case)
        cell_based_gf = multiDict[scenario][country][metric,0][3,index[base]]
        imprArray[metric,0][3,:] = (imprArray[metric,0][3,:]-cell_based_gf)/cell_based_gf*100
      imprDict[scenario][country] = imprArray
    
  ## 3. Prepare the plot!
  fig, axs = plt.subplots(2,2, figsize=(7.14,6))#, sharex=True, sharey=True) # 16 cm of width
  # Add y label
  fig.text(0.04, 0.5, 'Kappa Standard improvement [%]', va='center', rotation='vertical')
  # Add x label
  fig.text(0.4, 0.04, 'RMSE improvement [%]', va='center')
  # Assig
  marker = {
    'cilp':'s',
    'fdi':'^',
    'wfdi':'X',
    'pd':'o'}
  alpha = {1:0.6, 2:0.3}
  for scenario in [1,2]:
    i=0
    j=0
    for m,metric in  enumerate(metricNames):
      for country in case_studies:
        l = 'K and '+metric.upper()
        axs[i,j].axvline(x=0, alpha=0.2)
        axs[i,j].axhline(y=0, alpha=0.2)
        axs[i,j].scatter(
          imprDict[scenario][country][m,0][1,[5]],
          imprDict[scenario][country][m,0][2,[5]],
          s = [200], #30,300
          marker = marker[metric],
          edgecolor=countryColors[country],
          facecolor='none',
          alpha=alpha[scenario])
        axs[i,j].set_title(l)
      j=+1
      if m==1:
        i=1
        j=0
      
  # Create three legends. One for countries, one for metrics, one for scenarios
  #labels: metricnames and case_studies and scenarios [1,2]
  # add blank color to the color list
  c1 = [ countryColors[c] for c in case_studies ]
  # Now add opacity
  c1 = [ colors.to_rgba(c1[c])[:-1]+(alpha[1],) for c in range(len(case_studies)) ]
  c3 = [ colors.to_rgba(countryColors[case_studies[0]])[:-1]+(alpha[1],) ]
  c4 = [ colors.to_rgba(countryColors[case_studies[0]])[:-1]+(alpha[2],) ]
  used_colors= c3+c4
  labels = ['scenario 1','scenario 2']
                                     
  # Make patches:
  patches = {
    0: [ mpatches.Patch(color=c1[j],
                              label="{l}".format(l=case_studies[j]) ) for j in range(len(case_studies)) ],
    1: [ mpatches.Patch(color=used_colors[j],
                              label="{l}".format(l=labels[j]) ) for j in range(len(labels)) ]}
  for i in [0,1]:
    # Set a legend's anchor
    leg = axs[0,i].legend(
      handles=patches[i],
      bbox_to_anchor=(0., 1.2, 1, .102),
      loc='lower center',
      ncol=len(patches[i]),
      mode="expand",
      borderaxespad=0.,
      bbox_transform=axs[0,i].transAxes,
      fontsize=6)
    leg.get_frame().set_edgecolor('darkviolet')
    leg.get_frame().set_linewidth(0.50)
    
  # Set the name and clear the directory if needed
  setNameClearSave('multiobjective_compared_to_'+base+'_based')


