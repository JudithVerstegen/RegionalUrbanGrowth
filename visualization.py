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
countryColors = plt.cm.rainbow(np.linspace(0,1,3))
country_colors = {'IE':'mediumspringgreen','IT':countryColors[0],'PL':countryColors[2]}
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

def getResultsArrayEverySet(aim):
  results = np.empty((len(all_metrices),6,165))
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
        n_array = calibrate.getNormalizedArray(av_array,scenario,aim,kappa=k)
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
  scenario=1
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

  setNameClearSave('urb_changes_modelled')     

def plotBestPerformers(): #Done
  """
  Plot five subplots, each for a goal function.
  Each subplot shows boxplots for 15% of best parameter sets, for 3 case studies and 2 scenarios
  """
  nrOfBestPerformers = 25 #~15% of all parameters
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(5, figsize=(8,8),sharex=True) # 16 cm of width
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  # Add y label
  fig.text(0.04, 0.5, "distribution of parameters for "+str(nrOfBestPerformers)+' best solutions', va='center', rotation='vertical')
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
      c = country_colors[country]
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
        # Set y label
        if metric == 'kappa':
          axs[i].set_ylabel(metric[0].upper())
        else:
          axs[i].set_ylabel(metric.upper())
        # Plot boxplot
        bp = axs[i].boxplot(
          y,
          whis=[5,95],# set the whiskers at specific percentiles of the data
          widths=width,
          positions=positionDict[j],
          patch_artist=True)
        # Assign colors and transparency to boxplots and fliers
        for box in bp['boxes']:
          box.set_facecolor(c_new)
          #box.set_alpha(0.5)
        for flier in bp['fliers']:
          flier.set(marker='o', color=c_new, alpha=0.5, markersize=2)
        j+=1
        if m_i==0:
          boxplots.append(bp)
    i+=1
  # Set the ticks and ticklabels for all axes
  plt.setp(axs, xticks=ind, xticklabels=drivers)#,yticks=[1, 2, 3])
  for ax in axs:
    ax.set_xlim(0.4, 4.6)
  # Create a legend:
  leg = ax.legend([ box['boxes'][0] for box in boxplots ],
            [x+str(y) for x in case_studies for y in [1,2]],
            bbox_to_anchor=(0., 1.02, 1, .102),
            loc='lower center',
            ncol=6,
            mode="expand",
            bbox_transform=axs[0].transAxes,
            borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  
  setNameClearSave(str(nrOfBestPerformers)+'_top_performers')

def plotGoalFunctionEverySet(): #DONE
  """
  Plot the values for a goal function durnig calibration. Values for every parameter set tested
  5 subplots stacked vertically, one for every metric
  """

  # First, get all results
  results = getResultsArrayEverySet('calibration')
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
                 c = country_colors[country])
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

def plotParameterValues(): #
  """
  Plot bars presenting parameters (0-1, y axis) for 5 goal funcstions (x axis)
  There are 5 metrics, 3 case studies, 2 scenarios --> 15 bars
  """
 
  # Create figure
  fig,ax = plt.subplots(1,figsize=(8,4))
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
  plt.xticks(ind, [x.upper() for x in metricNames] + ['K'])
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


  axs={}
  #Assign positions of bars for each scenario:
  positions = {
    1:np.array([[i - (2.5*width+space),i - 0.5*width,i + (1.5*width+space)] for i in ind ]).flatten(),
    2:np.array([[i - (1.5*width+space),i + 0.5*width,i + (2.5*width+space)] for i in ind ]).flatten()}
  # Now plot the bars. For each scenario, the bars are plotted as a different ax
  for scenario in [1,2]:
    bottom=0
    axs[scenario] = []
    for driver in range(len(drivers)):
      label = drivers[driver]
      if scenario==2:
        labels=None
      axs[scenario].append(ax.bar(
        positions[scenario],
        p_dict[scenario][driver],
        width=width,
        bottom=bottom,
        color=driverColors[driver],
        alpha=alpha[scenario],
        label=label))
      bottom = bottom + np.array(p_dict[scenario][driver])

  # Draw lines dividing scenario bars and add annotation with the country symbol
  for xtick in [0,1,2,3,4]:
    for c,country in enumerate(case_studies):
      p = xtick-2*width-space+c*(2*width+space)
      plt.axvline(p, alpha=0.5,c='white',linestyle='--', linewidth=0.5)
      plt.annotate(country,xy=(p,1.01), rotation=0, color="darkviolet",ha='center',weight='bold')

  #Create a legend
  legend1 = plt.legend([axs[1][0],axs[2][0]], ['scenario 1','scenario 2'], loc=1)
  leg = ax.legend(
    axs[1],
    drivers,
    bbox_to_anchor=(0, 1.05, 1, .102),
    loc='lower center',
    ncol=4,
    mode="expand",
    bbox_transform=ax.transAxes,
    borderaxespad=0.)
  leg.get_frame().set_edgecolor('darkviolet')
  leg.get_frame().set_linewidth(0.50)
  plt.gca().add_artist(legend1)
  # Set the name and clear the directory if needed
  setNameClearSave('parameters', scenario=None)
plotParameterValues()




def plotNonDominatedSolutions(metrics,aim):
  # Plot all the validation metrics for all parameter sets, for a case study, both scenarios
  # Plot will be used for multi-objective calibration WITHOUT CILP
  results = {
    'calibration':createNormalizedResultDict(metrics,'calibration'),
    'validation':createNormalizedResultDict(metrics,'validation')}

  fmt = {1:'-o',2:'--o'}
  # Create the figure
  fig = plt.figure(figsize=(6.4,3.2)) # Figure size set to give 16 cm of width
  plt.ylabel('normalized metric value')       
  plt.xlabel('metric')
  
  for country in case_studies:
    for scenario in [1,2]:
      #non_dominated = is_pareto_efficient_simple(results[scenario])
      asum = np.reshape(np.sum(results['calibration'][country][scenario],axis=1),(165,1))
      #non_dominated = is_pareto_efficient_simple(asum)
      non_dominated = is_pareto_efficient_simple(results['calibration'][country][scenario])
      aLabel = country+str(scenario)
      for n in range(165):
        if non_dominated[n] == True:        
          #c = 'black'
          lw=0.5
          m=1
        else:
          #c = country_colors[country]
          lw=0.01
          m=0
        if n>0:
          aLabel=None         
        plt.plot(metrics,results[aim][country][scenario][n,:],
           fmt[scenario],label=aLabel,
           linewidth = lw,
           c = country_colors[country],
           markersize=m,
           markerfacecolor=country_colors[country])
    
  leg = plt.legend(loc='lower right')
  for line in leg.get_lines():
    line.set_linewidth(1.0)
  #plt.show()

  wPath = os.path.join(os.getcwd(),'results', 'non_dominated_solutions_'+aim)
  if os.path.exists(wPath):
      os.remove(wPath)

  # Save plot and clear    
  plt.savefig(wPath, bbox_inches = "tight",dpi=300)
  plt.clf()
  
def plotNonDominatedSolutionsSmall(metrics):
  # Now create small plots for each country seperate
  results = createNormalizedResultDict(metrics,'calibration')
  fmt = {1:'-o',2:'--o'}
  for country in case_studies:
    fig = plt.figure(figsize=(2.1,2.1))   
    for scenario in [1,2]:
      non_dominated = is_pareto_efficient_simple(results[country][scenario])
      for n in range(165):
        if non_dominated[n] == True:        
          lw=0.5
          m=1
        else:
          lw=0
          m=0
        plt.plot(metrics,results[country][scenario][n,:],
           fmt[scenario],
           linewidth = lw,
           c = country_colors[country],
           markersize=m,
           markerfacecolor=country_colors[country])
    wPath = os.path.join(os.getcwd(),'results', 'non_dominated_solutions_'+country)
    if os.path.exists(wPath):
        os.remove(wPath)

    # Save plot and clear    
    plt.savefig(wPath, bbox_inches = "tight",dpi=300)
    plt.clf()
    plt.close()

def plotManyParetoSolutions(metrics):
  # Plot will be used for multi-objective function based on 1-4 functions
  # Get the normalized results of metrics (RMSE or Kappa), for every parameter set
  results = {
    'calibration':createNormalizedResultDict(metrics,'calibration'),
    'validation':createNormalizedResultDict(metrics,'validation')}

  fmt = {1:'-o',2:'-o'}
  # Get all the posible function combinations (manually):
  function_combinations = [
    [0],[1],[2],[3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,1,2,3]]
  labels = [0,4,10,14]
  ending={0:''}
  # Create the figure
  fig,axs = plt.subplots(1,1, squeeze=False,figsize=(6.4,3.2)) # Figure size set to give 16 cm of width
  plt.ylabel('normalized metric value')       
  plt.xlabel('metric')
  old_len=0
  plots=[]
  i=0
  for country in case_studies:
    for scenario in [1,2]:
      for f_i, f in enumerate(function_combinations):
        if f_i in labels and i<len(labels):
          aLabel = str(len(f)) + ' objective'+ending.get(i,'s')
          i+=1
        else:
          aLabel=None
        results_f = results['calibration'][country][scenario][:,f] # f (columns) = normalized values for selected metrics
        asum = np.reshape(np.sum(results_f,axis=1),(165,1))
        non_dominated = is_pareto_efficient_simple(asum)
        #non_dominated = is_pareto_efficient_simple(results['calibration'][country][scenario])
        for n in range(165):
          if non_dominated[n] == True:
            axs[0,0].plot(
              np.array(metrics)[f],
              results['validation'][country][scenario][n,f], # n(rows) contain results for given parameters
              fmt[scenario],
              label=aLabel,
              
              linewidth =0.2,
              c = functionColors[len(f)-1],
              markersize=3,
              markerfacecolor=functionColors[len(f)-1])
            
            '''axs[0,0].plot(
              np.array(metrics)[f],
              results[aim][country][scenario][n,f],
              'ro-')'''
            

  #leg.get_frame().set_edgecolor('darkviolet')
  #leg.get_frame().set_linewidth(0.50)


  leg= axs[0,0].legend(loc='lower right')
  # Set the name and clear the directory if needed
  plt.show()
  #setNameClearSave('non_dominated_number_of_objectives', scenario=None)





