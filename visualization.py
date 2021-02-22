# -*- coding: cp1252 -*-
import pickle
import os
import metrics
import numpy as np
import parameters
import pcraster as pcr
import calibrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import scipy.stats as stats
##from openpyxl import load_workbook
##from mpl_toolkits.axes_grid1.inset_locator import inset_axes

print("This is a script to plot figures presenting outputs of the urban growth model")

#########################
### Global variables ###
#########################

# Get metrics
metricNames = parameters.getSumStats()
locationalMetric = parameters.getLocationalAccuracyMetric()
all_metrices = metricNames+locationalMetric

all_optional_metrics = ['cilp','cohes','contag','ed','fdi','lpi','pd','wfdi'] \
    + locationalMetric


# Get case studies
country = parameters.getCountryName()
case_studies = parameters.getCaseStudies()
# Get calibration scenarios
scenarios = parameters.getCalibrationScenarios()
# Name the possible combinations of case studies and countries:
cases = [(country,scenario) for country in case_studies for scenario in scenarios]
# Get the actual case studies :)
cities = {case_studies[0]:'Dublin',case_studies[1]:'Milan',case_studies[2]:'Warsaw',}

# Get number of zones and parameters
numberOfZones = parameters.getNumberOfZones()
numberOfParameters = parameters.getNumberofIterations()

# Create colors for zones and parameters
zoneColors = plt.cm.rainbow(np.linspace(0,1,numberOfZones))
parameterColors = plt.cm.rainbow(np.linspace(0,1,numberOfParameters))
countryColors = {
  'IE':(51/256,200/256,142/256),
  'IT':(125/256, 50/256,203/256),
  'PL':(205/256,51/256,51/256)
  }
functionColors = plt.cm.rainbow(np.linspace(0,1,4))
solutionColors = {1: 'blue', 2: 'red', 3: 'green', 4: 'magenta'}

# Create colors for maps:
c_obs = ['crimson', 'whitesmoke','black','white'] 
c_mod = c_obs#['lavender', 'dimgrey', 'gold']
driverColors = [(64,64,64),(184,13,72),(43,106,108),(242,151,36)] # NEIGH, TRAIN, TRAVEL, LU
driverColors = [ (x[0]/256,x[1]/256,x[2]/256) for x in driverColors ]
# Get the observed time steps.
# Time steps relate to the year of the CLC data, where 1990 was time step 0.
obsTimeSteps = parameters.getObsTimesteps()
observedYears = [parameters.getObsYears()[y] for y in obsTimeSteps]
calValYears = parameters.getCalibrationPeriod()

# Path to the folder with the metrics stored
workDir = parameters.getWorkDir()
resultFolder = os.path.join(workDir,'results',country, 'metrics')

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


#################
### FUNCTIONS ###
#################

def clearCreatePath(path, name): #ok
  if not os.path.isdir(path):
    os.makedirs(path)
  # Create dir of a file
  wPath = os.path.join(path, name)
  if os.path.exists(wPath):
      os.remove(wPath)
  return wPath
  
def setNameClearSave(figName, scenario=None): #ok
  # Set the font type to readible for Adobe
  plt.rcParams['pdf.fonttype'] = 42
  plt.rcParams['ps.fonttype'] = 42
  # Set the name and clear the directory if needed
  if scenario is None:
    name = ''
  else:
    name = '_scenario_'+str(scenario)
  fig_dir = os.path.join(os.getcwd(),'results','figures_calibration_1990_2000')
  wPath1 = clearCreatePath(fig_dir, figName+name+'.png')
  wPath2 = clearCreatePath(fig_dir, figName+name+'.pdf')
  # Save plot and clear    
  plt.savefig(os.path.join(resultFolder,wPath1),
              bbox_inches = "tight", dpi=600)
  plt.savefig(os.path.join(resultFolder,wPath2),
              bbox_inches = "tight")
  plt.close('all')

def getAverageResultsArrayEverySet(aim, use_all): #ok
  # plot all optional metrics or just the ones used in calibration and validation
  if use_all:
    metrics_to_use = all_optional_metrics
  else:
    metrics_to_use = all_metrices
  
  results = np.empty((len(metrics_to_use),
                      len(case_studies)*len(scenarios),
                      numberOfParameters))
  for i,m in enumerate(metrics_to_use):
    j=0
    for country in case_studies:
      for scenario in scenarios:
        # an_array is an array of arrays (contains rmse/kappa arrays for every p set)
        if m == 'kappa':
          an_array = calibrate.getKappaArray(scenario,aim,case=country)
        elif m == 'Ks':
          an_array = calibrate.getKappaSimulationArray('validation',case=country)
        elif m == 'A':
          an_array = calibrate.getAllocationArray('validation',case=country)
        else:
          an_array = calibrate.calcRMSE(m,scenario,aim,case=country)
        # get the goal function value of the metric for every parameter set  
        av_array = calibrate.getAveragedArray(an_array,scenario,aim)
        results[i,j] = av_array
        j+=1
  return results

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
 
def saveNonDominatedPoints_to_excel(aim, solution_space, objectives): # For n objectives and 1 multiobjecive
  '''
  solution_space in ['all', 'nondominated']
  objectives in ['n_objectives','nd_solutions']'''
  
  scenario = scenarios[0]
  # Loop case studies:
  for i,country in enumerate(case_studies):
    # Get the validation results for n+1 points
    results_nd_n_1 = calibrate.get_ND_n_1(country, scenario, aim, solution_space, objectives)
    # Assign rows and columns names
    rows = [ 'P'+str(i+1) for i in range(len(results_nd_n_1)) ]
    cols = [ m for m in all_metrices ]
    # Assign a filename    
    filename = 'non_dominated_'+aim+'_results_'+country
    # Save as excel file
    saveArrayAsExcel(results_nd_n_1, filename,sheet_name = aim[:3]+'_results',row_names=rows, col_names=cols)

    ## Add a sheet with all the points
    # Get validation metric values for non-domintaed solutions
    r_nd_v = calibrate.get_ND(country, scenario, aim)    
    # Assign rows and columns names
    rows = [ 'solution '+str(i) for i in range(len(r_nd_v)) ]
    cols = [ m for m in all_metrices ]
    # Convert to dataframe
    df = pd.DataFrame(r_nd_v,
                      index = rows,
                      columns = cols)
    # Update the file with the new sheet for selected points
    appendArrayAsExcel(df, filename, sheet_name='non-dominated')

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
  
class ScalarFormatterForceFormat(ScalarFormatter):
  def _set_format(self):#,vmin,vmax):  # Override function that finds format to use.
    self.format = "%6.2f"  # Give format here


#######################################
########### PLOTS FUNCTIONS ###########
#######################################

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    '''# Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")'''

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im#, cbar

def annotate_heatmap(im, p_mask, data=None, valfmt="{x:.2f}",
                     textcolors=["white", "black"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the significance.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(p_mask[i,j])])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def func(x, pos):
  return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

def corrSuitabilityMaps():
  #### Script to find corelation between suitbility maps
  for country in case_studies:
    suitDir = os.path.join(os.getcwd(),'results','figures','suitability_maps')
    workDir = parameters.getWorkDir()
    resultFolder = os.path.join(workDir,'results',country, 'metrics')
    fig_dir = os.path.join(os.getcwd(),'results','figures')
      

    pointFile = 'sampPointNr.col'
    pointDir = os.path.join(os.getcwd(),'input_data',country, pointFile)
    file_list = os.listdir(suitDir)
    file_list = [x for x in file_list if x.startswith(country)]
    maps = [os.path.join(os.getcwd(),'results','figures','suitability_maps',f) for f in file_list]
    maps_pairs = [(file_list[i],file_list[j]) for i in range(len(file_list)) for j in range(i+1, len(file_list))]
    maps_pairs = [(x[8:],y[8:]) for x,y in maps_pairs]

    arrays = []
    for a_map in maps:
      arrays.append(metrics.map2Array(a_map,pointDir))

    pairs = [(arrays[i],arrays[j]) for i in range(len(arrays)) for j in range(i+1, len(arrays))]

    for i,pair in enumerate(pairs):
      # Check for NaNs
      mask = np.isnan(pair[0]) | np.isnan(pair[1])
      # Mask, to remove NaNs
      x = pair[0][~mask]
      y = pair[1][~mask]

      print(x.shape, y.shape)
      print(np.corrcoef(x,y))
      
      # Plot
      plt.scatter(x,y,s=0.1)
      a_string = 'correlation coefficient between '+\
                 maps_pairs[i][0]+' and '+maps_pairs[i][1]+': '+str(np.corrcoef(x,y)[0,1])
      plt.title(a_string)
               
      plt.xlabel(maps_pairs[i][0])
      plt.ylabel(maps_pairs[i][1])
      
      
      aPath = os.path.join(fig_dir, country+'_'+maps_pairs[i][0]+'_'+maps_pairs[i][1])
      plt.savefig(os.path.join(resultFolder,aPath))
      plt.close('all')
    print(country,'done')

##############################    
#### PLOTS FOR THE ARTICLE ###
##############################

### FIGURE 1 was made in ArcMap ###

### FIGURE 2 ###
def plotDemand():
  # Create the figure
  fig, axs = plt.subplots(1,2, figsize=(5,1.5), gridspec_kw={'width_ratios': [1,2]})
  # Adjust spaces
  plt.subplots_adjust(wspace=0.1)  
  
  # Get the data
  for ic, c in enumerate(case_studies):
    afile = os.path.join(os.getcwd(),'input_data',c,'make_demand_manual.xlsx')
    df  = pd.read_excel(io=afile, header=None,engine='openpyxl')
    d = df.iloc[0:obsTimeSteps[-1],[11]].to_numpy().flatten()
    d = np.array([d[x-1] for x in obsTimeSteps])
    d = d/100 # Change ha to km2
    #demand[ic] = d
    axs[0].plot(d, label = cities[c], c = countryColors[c])
    
    
    d_d = np.empty_like(d)
    for x in range(len(d)):
      if x != len(d)-1:
        d_d[x+1] = (d[x+1]-d[x])/d[0]*100 #[%]
    d_d[0] = 0
    axs[1].bar(
      np.array([0,1,2,3])+ic/4,
      d_d[1:],
      label = cities[c],
      width=0.2,
      color = countryColors[c])
    

  # Axes ranges and lines for cal/val
  axs0lims = axs[0].get_ylim()
  axs1lims = axs[1].get_ylim()
  axs[0].set_ylim(0, axs0lims[1])
  axs[0].set_xlim(0, axs[0].get_xlim()[1])
  axs0lims = axs[0].get_ylim()
  
  cal = calValYears.get(1).get('calibration')
  val = calValYears.get(1).get('validation')
  
  axs[0].set_xticks(ticks=range(len(d)))
  axs[0].set_xticklabels(observedYears)
  axs[0].vlines(cal, axs0lims[0], axs0lims[1], colors='grey', linestyles = 'dashed')
  axs[0].vlines(val, axs0lims[0], axs0lims[1], colors='grey', linestyles = 'dashed')
  axs[0].text(cal[0]-0.75, 200, 'cal')
  axs[0].text(val[0]-0.75, 200, 'val')
  
  # Put the y axis to the right
  axs[1].yaxis.set_label_position("right")
  axs[1].yaxis.tick_right()
  axs[1].set_xticks(np.array([0,1,2,3])+0.25)
  axs[1].set_xticklabels(['1990-2000','2000-2006','2006-2012','2012-2018'])
  axs[1].vlines([cal[0] - 0.25], axs1lims[0], axs1lims[1], colors='grey', linestyles = 'dashed')
  axs[1].vlines([val[0] - 0.25], axs1lims[0], axs1lims[1], colors='grey', linestyles = 'dashed')
  axs[1].text(cal[0]-0.9, -8, 'cal')
  axs[1].text(val[0]-0.9, -8, 'val')
  

  # Assign labels
  axs[0].set_ylabel("Urban areas [km$^2$]")
  axs[1].set_ylabel('Annual change [%]')
  # Plot legend on the top
  leg = axs[0].legend(
    bbox_to_anchor=(0., 1.1, 2.1, .102),
    ncol = 3,
    mode='expand',
    frameon = False)
  # Set name
  aname = 'Figure 2 Urban areas'
  setNameClearSave(aname,scenario=None)#, fileformat='png')

### FIGURE 3 ###
def plotNonDominatedSolutions_singlebar(solution_space, objectives, trade_off = False):
  '''
  Find non-dominated combinations of metric values (for each parameter set)
  Plot 1 plot fopr a country in scenario 1
  x and y axis show values of two metrics
  thrid metric value is presented with shape color

  Multi-objective approach parameters:
  solution_space: selection of points closest to the 'ideal' point in ['all', 'nondominated'] 
  objectives: recquirement for selecting the 'ideal' point in ['n_objectives','nd_solutions']
  '''
  # Only for scenario 1
  scenario = scenarios[0]
  ## Create the figure. 
  fig = plt.figure(constrained_layout=True, figsize=(4,6))#, sharex=True, sharey=True)
  # Apply gridspec
  spec = fig.add_gridspec(nrows=len(case_studies)*2,ncols=2,hspace=1.5,wspace=.5,
                          height_ratios=np.array([ [0.1,0.9] for i in case_studies]).flatten())
  # Arrange subplots:
  #plt.subplots_adjust(wspace=0.35, hspace=0.35)
  # Add y label
  fig.text(-0.05, 0.5, 'RMSE ('+all_metrices[1].upper()+')', va='center', rotation='vertical')
  # Add x label
  fig.text(0.5, -0.05, 'RMSE ('+all_metrices[0].upper()+')', ha='center')
  
  ## 1.
  # Get results of validation metrics for all cases 
  results = {
    'calibration':calibrate.getResultsEverySet('calibration'),
    'validation':calibrate.getResultsEverySet('validation')}

  ## 2.
  # For each case study create a selection (mask) of non-dominated solutions identified in calibration
  # and a selection of four solutions: for 3 objectives and one multi-objective
  for i,country in enumerate(case_studies): 
    idx = i
    i = i*2

    ## 4. Get calibration values
    # Get metric values
    v_c = results['calibration'][country][scenario]
    # Subset non-dominated calibration results
    r_nd_c = calibrate.get_ND(country, scenario, aim='calibration')
    # Create a mask to find the values selected in terms of 3 objectives and 1 multi-objective
    c_mask = calibrate.get_ND_n_1_masks(country, scenario, solution_space, objectives)
    
    ## 4. Get validation values
    # Get metric values
    v_v = results['validation'][country][scenario]
    # Subset non-dominated calibration results
    r_nd_v = calibrate.get_ND(country, scenario, aim='validation')
    
    ## 5. Get the scale of the values, taking into account both calibration and validation values
    # Join calibration and validation results for third metric
    r_nd_2 = np.concatenate((r_nd_c[:,2],r_nd_v[:,2]), axis = 0)
    
    # Normalize the third metric values to get the scale for the marker between 0 and 1
    scale = np.array(calibrate.getNormalizedArray(r_nd_2)) # 1=best, 0=worst
    if scenario==1:
      scale=1-scale # 1=worst, 0=best
    # Combine calibration and validation results to find the total min and max values
    r_nd_all = np.concatenate((r_nd_c, r_nd_v),axis =0)
    ## 10. Create the legend 
    # Create a colormap   
    a_cmap = colors.LinearSegmentedColormap.from_list(
      'custom',
      [(0, '#ccff44'), # best 
       # to much colors : (np.split(scale,2)[1].max()/2, '#00bbee'),
       (np.split(scale,2)[1].max(), '#000088'), # <== good ones
       (np.split(scale,2)[1].max()+0.000001, '#ffffff'), # white
       (np.split(scale,2)[0].min()-0.000001, '#ffffff'), # white
       (np.split(scale,2)[0].min(), '#cccc22'), # ==> bad ones
       (np.split(scale,2)[0].min()+(1-np.split(scale,2)[0].min())/2, '#cc1144'),
       (1,'#990033')], # worst
       N=350)
    # Normalize the colors
    a_norm= colors.Normalize(vmin=r_nd_2.min(), vmax=r_nd_2.max())
    # Create a colorbar
    a_bar = matplotlib.cm.ScalarMappable(
      norm = a_norm, 
      cmap = a_cmap)
    a_bar.set_array([])
    # Add a subplot with a full width
    c_ax = fig.add_subplot(spec[i,:])
    # Adjust the size of the colorbar to the size of the ax
    axins1 = inset_axes(c_ax, width="100%",height="50%",
                        bbox_to_anchor=(0.01, 0.2, 1, 1),
                        bbox_transform=c_ax.transAxes)
    c_bar = plt.colorbar(
      a_bar,
      cax=axins1,
      orientation='horizontal',
      label=country,
      cmap=a_cmap)
    
    # Add the label with the locational metric
    c_bar.set_label(all_metrices[2])
    # Move the label above the bar
    c_bar.ax.xaxis.set_label_position('top')
    # Hide the ax as we need the colorbar only :)
    c_ax.set_frame_on(False)
    c_ax.xaxis.set_visible(False)
    c_ax.yaxis.set_visible(False)    
    ## 7. Plot calibration and validation results
    i+=1
    
    # Loop calibration and validation values
    for j, r_nd in enumerate([r_nd_c, r_nd_v]):
      # Add an axis for all non dominated results
      ax_all = fig.add_subplot(spec[i,j])
      # Plot a scatter plot
      ax_all.scatter(
        r_nd[:,0],
        r_nd[:,1],
        marker='x',
        s=20,
        linewidth=0.5,
        c = a_cmap(np.split(scale,2)[j]),
        cmap=a_cmap,
        alpha=0.7,
        label="-".join([country,str(j)]))
      
      # Plot (again) the point with the minimum errors for 4 objectives
      s_r_nd = r_nd[c_mask]
      # Get the indices of the mask
      inx = [i for i, x in enumerate(c_mask) if x]
      # Add an axis for seected points
      ax_selected = fig.add_subplot(spec[i,j])
      # Plot the selected solutions
      ax_selected.scatter(
        s_r_nd[:,0],
        s_r_nd[:,1],
        s=100,
        linewidths=10,
        marker='x',
        c = a_cmap(np.split(scale,2)[j][inx]),
        cmap=a_cmap,
        alpha=0.9,
        label="-".join([country,str(j)]))
      
      ## 8. Print the number of the selected solution next to the point
      # Get the array of points
      selected_points = calibrate.get_ND_n_1(
        country,
        scenario,
        ['calibration','validation'][j],
        solution_space,
        objectives)
      # Plot the numbers
      for p,c in enumerate(selected_points):
        a_point = c
        ax_selected.text(a_point[0],
                         a_point[1],
                         'P'+str(p+1),
                         va='center',
                         ha='center')
      
      ## 9. Adjust the plot
      # Find the maximum values
      max0_r_nd, max1_r_nd, max2_r_nd = r_nd.max(axis=0)
      #print(country, j, 'max WFDI', r_nd.max(axis=0)[0], 'max COHESION', r_nd.max(axis=0)[1])
      # Find the minimum values
      min0_r_nd, min1_r_nd, min2_r_nd = r_nd.min(axis=0)
      #print(country, j, 'min WFDI', r_nd.min(axis=0)[0], 'min COHESION', r_nd.min(axis=0)[1])
      ### Remove the common scale:
      # Set the limits
      ax_all.set_xlim(left=min0_r_nd-(max0_r_nd-min0_r_nd)*0.1,right=max0_r_nd+(max0_r_nd-min0_r_nd)*0.1)
      ax_all.set_ylim(bottom=min1_r_nd-(max1_r_nd-min1_r_nd)*0.1,top=max1_r_nd+(max1_r_nd-min1_r_nd)*0.1)
      
      # Specify the number of ticks on both or any single axes
      ax_all.locator_params(tight=True, nbins=5)
      ax_all.locator_params(tight=True, nbins=5)
      # Print the number of the non-dominated solutions on the left subplot
      # Get the limits of the ax
      x0=ax_all.get_xlim()[0]
      x1=ax_all.get_xlim()[1]
      y0=ax_all.get_ylim()[0]
      y1=ax_all.get_ylim()[1]
      
      ### Remove the common scale:
      # create ticks positions for x,y, and z axis
      x_ticks = np.linspace(min0_r_nd, max0_r_nd, 3)
      y_ticks = np.linspace(min1_r_nd, max1_r_nd, 3)
      # Assign 4 ticks and labels to the x, y and z (colorbar)axis.
      ax_all.set_xticks(x_ticks)
      ax_all.set_xticklabels('%.4f' % x for x in x_ticks)
      ax_all.set_yticks(y_ticks)
      ax_all.set_yticklabels('%.4f' % y for y in y_ticks)

      # Assign the subplot names
      ids = ['a','b','c','d','e','f']
      # Print the subplot id
      ax_all.text(
        x1-(x1-x0)*.075,
        y1-(y1-y0)*.075,
        ids[i-1+j],
        fontsize=6,
        weight='bold')
        
      # Print the number of non-dominated solutions in the bottom left corner
      if j==0:
        ax_all.text(
          x0+(x1-x0)*.01,
          y0+(y1-y0)*.01,
          'n='+str(len(r_nd_c)),
          fontsize=6,
          weight='bold')
        # Print the case study name
        ax_all.text(
          x0-(x1-x0)*.25,
          y0+(y1-y0)*.5,
          cities[country],
          weight='bold',
          rotation = 'vertical')
      
  # Add tiles
  fig.text(0.28,-0.01,'CALIBRATION',ha='center',weight='bold')
  fig.text(0.78,-0.01,'VALIDATION', ha='center',weight='bold') 
    
  # Save plot and clear
  aname = 'Figure 3 Performance'
  setNameClearSave(aname)#, fileformat='png')

### Figure 3 - v2 ###

def plotNonDominatedSolutions_multibar(solution_space, objectives, trade_off = False):
  '''
  Find non-dominated combinations of metric values (for each parameter set)
  Plot 1 plot for a country in scenario 1
  x and y axis show values of two metrics
  thrid metric value is presented with color

  Multi-objective approach parameters:
  solution_space: selection of points closest to the 'ideal' point in ['all', 'nondominated'] 
  objectives: recquirement for selecting the 'ideal' point in ['n_objectives','nd_solutions']
  '''
  # Only for scenario 1
  scenario = scenarios[0]
  ## Create the figure. 
  fig = plt.figure(figsize=(5,6))#, sharex=True, sharey=True)
  # Apply gridspec
  spec = fig.add_gridspec(nrows=len(case_studies),ncols=3)#,hspace=1.5,wspace=.5,
                          #height_ratios=np.array([ [0.1,0.9] for i in case_studies]).flatten())

  
  ## 1.
  # Get results of validation metrics for all cases 
  results = {
    'calibration':calibrate.getResultsEverySet('calibration'),
    'validation':calibrate.getResultsEverySet('validation')}

  ## 2.
  # For each case study create a selection (mask) of non-dominated solutions identified in calibration
  # and a selection of four solutions: for 3 objectives and one multi-objective
  for i,country in enumerate(case_studies): 
    i = i
    print(i)

    ## 4. Get calibration values
    # Get metric values
    v_c = results['calibration'][country][scenario]
    # Subset non-dominated calibration results
    r_nd_c = calibrate.get_ND(country, scenario, aim='calibration')
    # Create a mask to find the values selected in terms of 3 objectives and 1 multi-objective
    c_mask = calibrate.get_ND_n_1_masks(country, scenario, solution_space, objectives)
    
    ## 4. Get validation values
    # Get metric values
    v_v = results['validation'][country][scenario]
    # Subset non-dominated calibration results
    r_nd_v = calibrate.get_ND(country, scenario, aim='validation')
    
    ## 5. Get the scale of the values, taking into account both calibration and validation values
    # Join calibration and validation results for third metric
    r_nd_2 = np.concatenate((r_nd_c[:,2],r_nd_v[:,2]), axis = 0)
    
    # Normalize the third metric values to get the scale for the marker between 0 and 1
    scale = np.array(calibrate.getNormalizedArray(r_nd_2)) # 1=best, 0=worst
    if scenario==1:
      scale=1-scale # 1=worst, 0=best
    # Combine calibration and validation results to find the total min and max values
    r_nd_all = np.concatenate((r_nd_c, r_nd_v),axis =0)

    
    ## 5. Plot calibration and validation results
    
    # Loop calibration and validation values
    for j, r_nd in enumerate([r_nd_c, r_nd_v]):
      print('j ' + str(j))
      # Add an axis for all non dominated results
      ax_all = fig.add_subplot(spec[i,j])
      # Plot a scatter plot
      scat = ax_all.scatter(
        r_nd[:,0],
        r_nd[:,1],
        marker='x',
        s=20,
        linewidth=0.5,
        c = r_nd[:,2],#a_cmap(np.split(scale,2)[j]),
        #cmap=a_cmap,
        alpha=0.7,
        label="-".join([country,str(j)]))
      
      # Plot (again) the point with the minimum errors for 4 objectives
      s_r_nd = r_nd[c_mask]
      # Add an axis for seected points
      # Plot the selected solutions
      ax_all.scatter(
        s_r_nd[:,0],
        s_r_nd[:,1],
        s=60,
        linewidths=6,
        marker='x',
        c = s_r_nd[:,2], #a_cmap(np.split(scale,2)[j][inx]),
        #cmap=a_cmap,
        alpha=0.9,
        label="-".join([country,str(j)]))
      
      ## 6. Print the number of the selected solution next to the point
      # Get the array of points
      selected_points = calibrate.get_ND_n_1(
        country,
        scenario,
        ['calibration','validation'][j],
        solution_space,
        objectives)
      # Plot the numbers
      for p,c in enumerate(selected_points):
        a_point = c
        ax_all.text(a_point[0],
                    a_point[1],
                    'P'+str(p+1),
                    va='center',
                    ha='center')
     
        
      ## 7. Adjust the plot
      # # Find the maximum values
      # max0_r_nd, max1_r_nd, max2_r_nd = r_nd.max(axis=0)
      # #print(country, j, 'max WFDI', r_nd.max(axis=0)[0], 'max COHESION', r_nd.max(axis=0)[1])
      # # Find the minimum values
      # min0_r_nd, min1_r_nd, min2_r_nd = r_nd.min(axis=0)
      # #print(country, j, 'min WFDI', r_nd.min(axis=0)[0], 'min COHESION', r_nd.min(axis=0)[1])
      # ### Remove the common scale:
      # # Set the limits
      # ax_all.set_xlim(left=min0_r_nd-(max0_r_nd-min0_r_nd)*0.1,right=max0_r_nd+(max0_r_nd-min0_r_nd)*0.1)
      # ax_all.set_ylim(bottom=min1_r_nd-(max1_r_nd-min1_r_nd)*0.1,top=max1_r_nd+(max1_r_nd-min1_r_nd)*0.1)
      
      # Specify the number of ticks on both or any single axes
      ##ax_all.locator_params(tight=True, nbins=4)
      ##ax_all.locator_params(tight=True, nbins=4)
      # Print the number of the non-dominated solutions on the left subplot
      # Get the limits of the ax
      x0=ax_all.get_xlim()[0]
      x1=ax_all.get_xlim()[1]
      y0=ax_all.get_ylim()[0]
      y1=ax_all.get_ylim()[1]
      
      # ### Remove the common scale:
      # # create ticks positions for x,y, and z axis
      # x_ticks = np.linspace(min0_r_nd, max0_r_nd, 3)
      # y_ticks = np.linspace(min1_r_nd, max1_r_nd, 3)
      # # Assign 4 ticks and labels to the x, y and z (colorbar)axis.
      # ax_all.set_xticks(x_ticks)
      # ax_all.set_xticklabels('%.4f' % x for x in x_ticks)
      # ax_all.set_yticks(y_ticks)
      
      xfmt = ScalarFormatterForceFormat()
      xfmt.set_powerlimits((0,0))
      ax_all.xaxis.set_major_formatter(xfmt)
      
      yfmt = ScalarFormatterForceFormat()
      yfmt.set_powerlimits((0,0))
      ax_all.yaxis.set_major_formatter(yfmt)

      # # Assign the subplot names
      # ids = ['a','b','c','d','e','f']
      # # Print the subplot id
      # ax_all.text(
      #   x1-(x1-x0)*.075,
      #   y1-(y1-y0)*.075,
      #   ids[i-1+j],
      #   fontsize=6,
      #   weight='bold')
        
      # Print the number of non-dominated solutions in the bottom left corner
      if j==0:
        ax_all.text(
          x0+(x1-x0)*.01,
          y0+(y1-y0)*.01,
          'n='+str(len(r_nd_c)),
          fontsize=6,
          weight='bold')
        # Print the case study name and y-label
        ax_all.set_ylabel(cities[country] + '\n' + 'RMSE ('+all_metrices[1].upper()+')')

      if i == 2:
        # print x-label
        ax_all.set_xlabel('\n\n\n\nRMSE ('+all_metrices[0].upper()+')')
      
      ## 8. Colorbar 
      yfmt = ScalarFormatterForceFormat()
      yfmt.set_powerlimits((0,0))
      c_bar = plt.colorbar(scat, ax=ax_all, orientation='horizontal',
                           pad=0.2,
                           ticks=[np.min(r_nd[:,2]), np.max(r_nd[:,2])],
                           format=yfmt)   
      #c_bar.formatter.set_powerlimits((0, 0))
      # Add the label with the locational metric
      c_bar.set_label(all_metrices[2], rotation=0, labelpad=-4) 
      
    ## 9. Add Spearman rank correlation values
    scenario=scenarios[0]
    # Create an array to store the Spearman rho value
    spearmanMatrix = np.empty((len(all_metrices), 1))
    # Create an array to store p_value mask
    p_mask = np.zeros((len(all_metrices), 1))
    for m, metric in enumerate(all_metrices):
      rho, p_val = stats.spearmanr(r_nd_c[:,m],r_nd_v[:,m])
      # Fill the matrix with correlation between the values, if p_val < 0.001
      spearmanMatrix[m,0] = rho
      # Fill the significant mask based no p_value
      if p_val<0.01:
        p_mask[m,0] = 1
    print(cities[country])
    print(spearmanMatrix)
    ##print(p_mask)
    # Plot the heatmaps for every case study
    ax = fig.add_subplot(spec[i,2])
    im = heatmap(spearmanMatrix, all_metrices, [''], ax=ax,
                cmap="PRGn", vmin=-1, vmax=1,cbarlabel="Spearman coeff.")
    # Add coeff. values. The ones with p_val > 0.01 are black, other are white
    # "The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so."
    # from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    # annotate_heatmap(im, p_mask, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
    annotate_heatmap(im, np.ones((len(all_metrices), 1)), 
                     valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
    
    if i == 2:
      #cbar.set_label("\n\n\n\nSpearman coeff.", rotation=0, va="bottom", labelpad=4)
      ax.set_xlabel("\n\n\n\nSpearman coeff.")
    
    i+=1
      
  # Add tiles
  fig.text(0.23,0.9,'CALIBRATION\n',ha='center',weight='bold')
  fig.text(0.51,0.9,'VALIDATION\n', ha='center',weight='bold') 
  fig.text(0.78,0.9,'SPEARMAN RANK CORR. \nCALIBRATION-VALIDATION', ha='center',weight='bold') 
  
  # Arrange subplots:
  plt.subplots_adjust(bottom=-0.1, wspace=0.3, hspace=0.2)

    
  # Save plot and clear
  aname = 'Figure 3 Performance'
  setNameClearSave(aname)



### FIGURE 4 and 7 ###                   
def plotWeights(solution_space, objectives, trade_off = False):
  """
  Plot bars presenting weights (0-1, y axis) for each selected point (two bars for point P4)
  for goal functions (x axis) for each case study

  Multi-objective approach parameters:
  solution_space: selection of points closest to the 'ideal' point in ['all', 'nondominated'] 
  objectives: requirement for selecting the 'ideal' point in ['n_objectives','nd_solutions']
  """
  scenario=scenarios[0]
  print('scenario',scenario)
  # Change the case studies order
  case_studies_ordered = [case_studies[0],case_studies[2],case_studies[1]]
  
  # Set the params
  drivers = ['NEIGH', 'TRAIN', 'TRAVEL', 'LU']
  parameterSets = calibrate.getParameterConfigurations()

  # Create the figure
  fig, axs = plt.subplots(1,len(case_studies), figsize=(4,2), sharey=True)
  # Set the x label
  axs[1].set_xlabel('Selected solutions')
  # Set the y label
  axs[0].set_ylabel('Weights')

  for c,country in enumerate(case_studies_ordered):
    ## Get the data for a country
    # Get the list of the indices of the selected non dominated solutions
    nd_points_indices = calibrate.get_ND_n_1_indices(country, scenario, solution_space, objectives)
    # Get also the actual averaged weights from n solutions
    ## Get the indices of the selected solutions to find the averaged structure
    the_indices = {
      'n_objectives': calibrate.get_ND_n_indices(country, scenario),
      'nd_solutions': calibrate.get_ND_indices(country, scenario)
      }
    # Get average weights
    n_weights_av = calibrate.getAverageWeights(the_indices[objectives])
    # Get the number of optimal solutions
    ind = np.arange(len(nd_points_indices)+1)    # the x locations for the groups
    # Fill the data for each point
    ## Create an empty dict to store weights for each driver. Drivers will be keyes:
    weights = {}
    ## Loop drivers
    for driver in range(len(drivers)):
      #Create a list to store the weights
      a_list=[]
      # Get the weights for n objectives (without the last point)
      for i,index in enumerate(nd_points_indices[:-1]):
        a_list.append(parameterSets[index][driver])
      # Add the averaged weights
      a_list.append(n_weights_av[driver])
      # Add the last point (P4)
      a_list.append(parameterSets[nd_points_indices[-1]][driver])
      # Save the weihts in the dictionairy
      weights[driver] = a_list

    # Plot the bars
    bottom=0
    for driver in range(len(drivers)):
      label = {1:drivers[driver], 2:None}
      axs[c].bar(
        ind,
        weights[driver],
        bottom=bottom,
        color=driverColors[driver],
        label=drivers[driver])
      
      bottom = bottom + np.array(weights[driver])
    # Set the ticks on the x axis
    axs[c].set_xticks(ind)
    axs[c].set_xticklabels(['P1','P2','P3','av','P4'])
    # Set the title of the sublot as the city (case study) name
    axs[c].set_title(cities[country])
    # Print the exact values of averaged weights
    bar = 0
    for d in n_weights_av:
      # Put text on the plot
      d = round(d,2)
      axs[c].text(ind[-2],bar+d/2,d, fontsize = 4, ha = 'center', va = 'center')
      bar = bar + d

  #Create a legend
  handles, labels = axs[0].get_legend_handles_labels()
  patch = handles[0][0]
  # plot the driver in the right top corner. Reverse the order to adjust tp the plot:
  leg1 = axs[1].legend(
    handles[::-1],
    labels[::-1],
    bbox_to_anchor=(0.5,-.2),
    loc='upper center',
    ncol=len(drivers),
    title='Drivers')
  leg1.set_frame_on(False)
   
  # Set the name and clear the directory if needed
  a_period = {1:'calibration', 2:'validation'}
  a_figure = {1:'Figure 4', 2:'Figure 7'}
  aname = a_figure[scenario] + ' Structure on '+ a_period[scenario] + ' period'
  setNameClearSave(aname, scenario=None)#, fileformat='png')

### FIGURE 5 and 6 ###  
def plotUrbanChanges(solution_space, objectives): #DONE
  """ 
  Plots observed and modelled changes for calibration/validatio period for each solution
  First column on the left presents the observed chages

  Multi-objective approach parameters:
  solution_space: selection of points closest to the 'ideal' point in ['all', 'nondominated'] 
  objectives: recquirement for selecting the 'ideal' point in ['n_objectives','nd_solutions']
  """
  scenario = scenarios[0]
  a_proces = {1:'validation',2:'calibration'}
  print('Urban changes in '+a_proces[scenario]+' period')
  # Get the number of solutions
  inx = calibrate.get_ND_n_1_indices(case_studies[0], scenario, solution_space, objectives)
  # Get the number of optimal solutions
  ind = len(inx)   # the x locations for the groups
  # Get the names of the solutions
  solutions = ['P1','P2','P3','P4']
    
  ## 1. Prepare the plot  
  fig, axs = plt.subplots(len(case_studies),ind+1, sharex=True, sharey=True, figsize=(7,4)) # 16 cm of width
  # Adjust the spaces
  plt.subplots_adjust(wspace=0.05, hspace=0.05)
  # Set colors
  colorsModelledChange = {-2: c_mod[0],
                          -1: c_mod[0],
                          0: c_mod[1],
                          1: c_mod[2],
                          2: c_mod[2]}
  
  ## 2. Loop and plot modellled countries
  # Select the time steps
  period = parameters.getCalibrationPeriod()[scenario]['validation'] # scenario 1 validation: [3,4]
  selected_time_steps = np.array(parameters.getObsTimesteps())[period] #[1,11,17,23,29]
##  years = [
##    str(np.array(observedYears)[period[0]-1]),
##    str(np.array(observedYears)[period[0]]),
##    str(np.array(observedYears)[period[1]]) ]

  years = [
    str(np.array(observedYears)[period[0]-1]),
    str(np.array(observedYears)[period[0]])]
  print('observed years:',years)
  
  i=0
  
  for country in case_studies:
    # Create an empty list to store observed maps
    obs_maps = []
    # Get the observed changes for each observed time step defined in 'period'
    for x in np.arange(len(period)+1):
      # Add the map for the observed change! Get the observed maps from CLC data:
      obs_maps.append(os.path.join(os.getcwd(),'observations',country, 'urb'+years[x][-2:]))
      print(country,os.path.join(os.getcwd(),'observations',country, 'urb'+years[x][-2:]))
      # Red the PCRaster format
      obs_maps[x] = pcr.readmap(obs_maps[x])
      # Convert the maps to the numpy arrays
      obs_maps[x] = pcr.pcr2numpy(obs_maps[x],99)
      # Change the arrays type to float to enable nan
      obs_maps[x] = obs_maps[x].astype('float')
      # Change the nodata cells to nans
      obs_maps[x][obs_maps[x] == 99] = np.nan
## UNCOMMENT AND ADAPT FOR MORE THAN ONE TIME STEP IN CALIBRATION
##    # Create the transition matrix between first two times steps
##    d_obs_0 = np.subtract(obs_maps[1], obs_maps[0])
##    # Create the transition matrix between first two times steps
##    d_obs_1 = np.subtract(obs_maps[2], obs_maps[1])
##    # Add the changes to get the total transition
##    d_obs = np.add(d_obs_0,d_obs_1)
    # Create the transition matrix between two times steps
    d_obs = np.subtract(obs_maps[1], obs_maps[0])
    # Get the unique values
    u_obs = np.unique(d_obs[~np.isnan(d_obs)])
    # Create a colormap for every unique value
    cmapL = [colorsModelledChange[v] for v in u_obs]
    cmap_obs = colors.ListedColormap(cmapL)
    # Plot the observed changes
    axs[i,0].imshow(d_obs, cmap=cmap_obs)
    axs[i,0].set(xlabel='observed', ylabel=cities[country])
    
    # Loop metrics and for each metric get the change map for seected country, and selected metric (goal function)    
    j=1
    # First, get the solutions
    n_1_indices = calibrate.get_ND_n_1_indices(country, scenario, solution_space, objectives)
    
    for m, inx in enumerate(n_1_indices):
      # Create an empty list to store observed maps
      mod_maps = []
      # Get the modelled changes between the observed time step defined in 'period' and start of the simulation
      for k in np.arange(len(period)):
        # Add the map for the observed change! Get the observed maps from CLC data:
        mod_array = calibrate.getModelledArray(
          'urb_subset_'+str(selected_time_steps[k]),case=country)
        mod_array = mod_array[0,inx,1]
        mod_maps.append(mod_array)
        # Reshape it into a size of a map
        mod_maps[k] = np.reshape(mod_maps[k], (1600,1600))
## UNCOMMENT AND ADAPT FOR MORE THAN ONE TIME STEP IN CALIBRATION
##      # Calculate the change between modelled and observed
##      d_mod_0 = np.subtract(mod_maps[0], obs_maps[0])
##      d_mod_1 = np.subtract(mod_maps[1], obs_maps[1])
##      # Add the changes to get the full transition
##      d_mod = np.add(d_mod_0, d_mod_1)
      d_mod = np.subtract(mod_maps[0], obs_maps[0])
      # Find states in the map to adjust the colors:
      u = np.unique(d_mod[~np.isnan(d_mod)])
      cmapL = [colorsModelledChange[v] for v in u]
      cmap = colors.ListedColormap(cmapL)
      # Plot
      axs[i,j].imshow(d_mod, cmap=cmap)
      axs[i,j].set(xlabel=solutions[m], ylabel=cities[country])
      
      j+=1
    i+=1

  # Adjust ticks and add background
  for ax in axs.flat:
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.xticks = ([])
    ax.label_outer()
    ax.set(facecolor=c_obs[3])

  # Create two legends. One for observed, one for modelled
  labels = {0:['urban to non-urban','no change','non-urban to urban', 'no data']}
  #1:['Modelled:','urban to non-urban','no change','non-urban to urban']}
  for i,c in enumerate([c_obs]):#,c_mod]):
    # Make patches:
    patches = [ mpatches.Patch(color=c[j], label="{l}".format(l=labels[i][j]) ) for j in range(len(labels[i])) ]
    # Create a legend
    leg = axs[0,0].legend(
      handles=patches,
      title = 'Urban areas transitions: '+years[0]+' to '+years[1],
      bbox_to_anchor=(0., 1.3, 5+5*0.1, .102),
      ncol = 4,
      mode="expand",
      fontsize=6)
    
  # Set name
  aname = ['Figure 6 Validation maps','Figure 5 Calibration maps']
  # Use this usually:
  setNameClearSave(aname[scenario-1],scenario=None)#, fileformat='png')

### FIGURE X (additional) ###
def plotMetrics(country, thisMetric):
  """
  # Plots two subplots for each metric
  # Plot the observed values of metrics in one subplot
  # and the modelled metric values for the calibrated parameters on the other subplot
  # All landscape metrics = 4 plots, 2 subplots each, stacked vertically
  # For FDI and WFDI plots values for zones on the diagonal of the case study area (zones: 0,5,10,15)
  """
  scenario = scenarios[0] 
  ## 1. Create the figure
  fig, axs = plt.subplots(4,4, sharex=True, figsize=(4,4))
  # Adjust the plot
  #plt.subplots_adjust(wspace=0.05, hspace=0.05)

  # Set linestyle for each scenario
  linestyle={'obs':'o-','mod':'o--'}

  # Add title
  fig.suptitle(str(thisMetric).upper())
  
  for m,metric in enumerate([thisMetric]):
    # Get the observed values of a metric
    zonesObserved = calibrate.getObservedArray(metric,case=country)
    zonesModelled = calibrate.getModelledArray(metric,scenario,case=country)
    indexCalibratedSet = calibrate.getCalibratedIndeks(metric,scenario,case=country)
    nd_n_1_indices = calibrate.get_ND_n_1_indices(country, scenario, 'all', 'n_objectives')

    ## 3. Plot the values        
    # Select the zones for plottong. Some metric are calculted for 16 zones. Other for one zone only:
    zonesNumber = len(zonesModelled[0][0][1])
    selected_zones = range(zonesNumber)
      
    # Loop the zones and the flattend axes:
    for z, ax in enumerate(axs.ravel()):
      # Plot observed values. Create and array and fill with values for each observed time step:
      metricValues = []
      for year in range(len(obsTimeSteps)):
        metricValues.append(zonesObserved[year][0][1][z][0])
      ax.plot(
        observedYears,
        metricValues,
        linestyle['obs'],
        linewidth = 0.7,
        markersize=1,
        label = cities[country] + ' observed',
        c='k')
      
      # Plot modelled values for each selected solution.
      for an_i, an_index in enumerate(nd_n_1_indices):
        # Create and array and fill with values for each time step:
        modelledValues = []
        for year in range(len(zonesModelled[:,0])):
          modelledValues.append(zonesModelled[year,an_index,1][z][0])
        ax.plot(
          np.arange(observedYears[0],observedYears[-1:][0]+1),
          modelledValues,
          linestyle['mod'],
          linewidth = 0.7,
          markersize=0,
          label = 'P'+str(an_i+1),
          c=solutionColors[an_i+1])

  # Iterate axes and add labels
  for zone, a in enumerate(axs.flat):
    # Add a zone number
    a.text(observedYears[0],ax.get_ylim()[1],'zone '+str(zone+1), fontsize=6)
    # Adjust x ticks
    a.set_xticks(observedYears[::2])
    a.set_xticklabels([str(x)[-2:]+"'" for x in observedYears[::2]])
    # Adjust y ticks
    plt.locator_params(axis='y', nbins=3)
    #a.set_yticks(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],3))
    #a.set_yticklabels(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],3))

  # Create a legend
  leg = axs[0,0].legend(
    bbox_to_anchor=(0., 1.35, 4.5, .102),
    ncol = 5,
    mode="expand",
    fontsize=6)
  
  # Set the name and clear the directory if needed
  setNameClearSave('Figure X Unscaled metrics '+country+'_'+str(thisMetric))

### FIGURE X (additional) ###
def plotAllocationDisagreement(country):
  scenario=scenarios[0]
  # Create the figure
  plt.figure(figsize=(5,2))
  # Get the data
  ad_array = calibrate.getAllocationArray(case=country)
  # Get the indices
  nd_n_1_indices = calibrate.get_ND_n_1_indices(country, scenario, 'all', 'n_objectives')
  # Plot A values for each solution
  for an_i, an_index in enumerate(nd_n_1_indices):
    plt.plot(ad_array[:,an_index],
             label = 'P'+str(an_i+1),
             c = solutionColors[an_i+1])
  plt.legend()
  plt.xticks(ticks=np.arange(len(observedYears)), labels=observedYears)
  plt.ylabel('Allocation Disagreement')
  plt.xlabel('Observed years')
  
  # Set name
  aname = 'Figure Xc Allocation Disagreement ' + country
  setNameClearSave(aname,scenario=None, fileformat='png')


### FIGURE X (additional) ###
def plotGoalFunctionEverySet(use_all): #DONE
  """
  Plot the values for a goal function durnig calibration.
  Values for every parameter set tested, plotted in subplots stacked vertically,
  one for every metric
  """

  # Get the metrics, all or only the ones used in the goal function
  if use_all:
    calibration_metrices = all_optional_metrics
    metric_units = [ 'RMSE/std' for i in 
                    ['cilp','cohes','contag','ed','fdi','lpi','pd','wfdi'] ] + locationalMetric
  else:    
    calibration_metrices = [ m.upper() for m in metricNames ] + locationalMetric
    metric_units = [ 'RMSE/std' for i in metricNames ] + locationalMetric
    
  # First, get all results
  results = getAverageResultsArrayEverySet('calibration', True)
  # Loop the data for all metrics to get minimum and maximum goal function values
  limits = {}
  # Get metric stats
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
  n = len(calibration_metrices)# number of subplots
  
  ## 2. Prepare the plot  
  fig, axs = plt.subplots(n, 2, figsize=(6,8), sharex = 'col')
  xticks = np.arange(0, parameters[-1]+10, 15.0)
  xticks=xticks.tolist()+[parameters[-1]]
  fig.align_ylabels()
  plt.subplots_adjust(hspace=0.4)
  #gf = ['$o_1$','$o_2$','$o_3$']
  gf = ['o('+c+')' for c in calibration_metrices]
  
  ## 3. Loop metrics. Each metric = new subplot
  for i,m in enumerate(calibration_metrices):
    j=0
    axs[i][1].set_ylabel(metric_units[i])
    axs[i][1].set_xticks(xticks)
    axs[i][1].set_xlim(0,max(parameters)+1)
    
    # Loop all the countries. Each suplot has data for all case studies:
    for country in case_studies:
      # Loop calibration scenarios:
      for scenario in scenarios:
        # set the min and max y axis values:
        axs[i][1].set_title(gf[i], pad=3)

        # standardize vaules by divinding by std
        stand_results = results[i,j]/np.std(results[i,j])
        
        amin = limits[m]['min']
        amax = limits[m]['max']
        axs[i][1].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        #axs[i].set_ylim([amin*0.9,amax*1.1])
        #axs[i].set_yticks([amin,amax])
        # Create the labels only for one metric
        if i>0:
          myLabel = {1:None,2:None}
        else:
          myLabel = {1:country+str(scenario),2:country+str(scenario)}
        fmt = {1:'-',2:'--'}
        # plot
        axs[i][1].plot(
          parameters,
          stand_results,
          fmt[scenario],
          linewidth = 0.5,
          label = myLabel[scenario],
          c = countryColors[country])
        plt.setp(axs[i][1].get_xticklabels(), rotation=90)
##        # Plot a line showing mean metric vaue in the parameter space and the value
##        axs[i].axhline(y=limits[m]['mean'], alpha=0.2,c='black',linestyle='--', linewidth=0.8)
##        axs[i].text(axs[i].get_xlim()[1]+1,
##                    limits[m]['mean'],
##                    'mean = '+str(np.format_float_scientific(limits[m]['mean'],precision=2)),
##                    fontsize=6,
##                    va='center')
        j+=1
  axs[i][1].set_xlabel('parameter set')

  ## 4. Loop metrics again, now plot over time. Each metric = new subplot
  calibration_metrices.pop(-1)
  for i,m in enumerate(calibration_metrices):
    j=0
    print(m)
    axs[i][0].set_ylabel(m)
    axs[i][0].set_xlim(min(observedYears)-2,max(observedYears)+2)
    
    # Loop all the countries. Each suplot has data for all case studies:
    for country in case_studies:
      # Loop calibration scenarios:
      for scenario in scenarios:

        # get results
        observed = calibrate.getObservedArray(m, case=country)
        metric_values = []
        for year in range(len(obsTimeSteps)):
            metric_values.append(observed[year][0][1])
        
        metric_values = np.array(metric_values)[:, :, 0].transpose()
        metric_values = metric_values[~np.isnan(metric_values).any(axis=1)]
        
        print(metric_values.shape)
        #axs[i].set_ylim([amin*0.9,amax*1.1])
        #axs[i].set_yticks([amin,amax])
        # plot
        c = countryColors[country]
        axs[i][0].boxplot(
            metric_values,
            positions = np.array(observedYears)+j-1,
            whis = [0,100],
            widths = 1,
            patch_artist=True,
            boxprops=dict(facecolor='w', color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c))
        j+=1
    axs[i][0].set_xticks(observedYears)

  axs[i+1][0].set_visible(False)  
  axs[i][0].set_xticklabels([str(x) for x in observedYears])
  axs[i][0].tick_params(labelbottom=True)
  axs[i][0].set_xlabel('time')

  
  # Create the legend
  leg = fig.legend(
    bbox_to_anchor=(0.28, 0.12),
    loc='lower center',
    ncol=len(case_studies),
    borderaxespad=0.,
    frameon = False,
    fontsize=9)
  
  # Set the name and clear the directory if needed
  setNameClearSave('Figure X Goal functions values - Values for each case study and metric (every plotted line) are divided by their std', scenario=None)#, fileformat='png')
  #setNameClearSave('Figure X Goal functions values', scenario=None)#, fileformat='png')

### FIGURE X (additional) ###    
def getSpearmanrResult():
  """ Plots correlation matrices for Spearman rank corelation
      for metric values for calibration and validation periods """
  scenario=scenarios[0]
  # Create an array to store the Spearman rho value
  spearmanMatrix = np.empty((len(all_metrices),len(all_metrices)))
  # Create an array to store p_value mask
  p_mask = np.zeros((len(all_metrices),len(all_metrices)))
  # Create the plot
  fig, axs = plt.subplots(1, 3, figsize=(9,2))
  # Get data for every country                            
  for i,c in enumerate(case_studies):
    # Get metric values for calibration
    results_nd_c = calibrate.get_ND(c, scenario, 'calibration')
    # Get metric values for validation
    results_nd_v = calibrate.get_ND(c, scenario, 'validation')
    # For every metric get the spearman rho (correlation coefficient) value
    for mx, metricx in enumerate(all_metrices):
      for my, metricy in enumerate(all_metrices):
        rho, p_val = stats.spearmanr(results_nd_c[:,mx],results_nd_v[:,my])
        # Fill the matrix with correlation between the values, if p_val < 0.001
        spearmanMatrix[mx,my] = rho
        # Fill the significant mask based no p_value
        if p_val<0.001:
          p_mask[mx,my] = 1
    # Plot the heatmaps for every case study
    im = heatmap(spearmanMatrix, all_metrices, all_metrices, ax=axs[i],
                 cmap="PRGn", vmin=-1, vmax=1,cbarlabel="Spearman coeff.")
    # Add coeff. values. The ones with p_val > 0.001 are black, other are white
    annotate_heatmap(im, p_mask, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
    # Add title
    axs[i].set_title(cities[c], pad=20)

  # Add a colorbar to the one last ax
  label="Spearman coeff."
  cbar = fig.colorbar(im, ax=axs.ravel().tolist(),cmap="Spectral")
  cbar.ax.set_ylabel(label, rotation=-90, va="bottom", fontsize=7)

  # Save!
  setNameClearSave('Figure X Spearman rank-order correlation coefficient', scenario=None)#, fileformat='png')

##########
## MAIN ##
##########
  
def main():
  # Setting used in the article:
  solution_space = 'nondominated'
  objectives = 'n_objectives'
  # Variables for testing:
##  country = 'IE'
  thisMetric = 'wfdi'
  aim='calibration'
  
##  print('Plotting...')
##  plotDemand()
##  print('Figure 2 plotted')
##  plotNonDominatedSolutions_multibar(solution_space, objectives, trade_off = False)
##  print('Figure 3 plotted')
##  plotWeights(solution_space, objectives, trade_off = False)
##  print('Figure 4 or 7 plotted')
  plotUrbanChanges(solution_space, objectives) # to remove? Not used, ArcGIS
  print('Figure 5 or 6 plotted')
##  plotMetrics(country, thisMetric)  # to remove? Not used, and strange layout now
##  print('Figure X plotted')
##  plotAllocationDisagreement(country)
##  print('Figure X plotted')
##  plotGoalFunctionEverySet(True)
##  print('Figure X plotted')
##  saveNonDominatedPoints_to_excel(aim, solution_space, objectives) # to remove? Not used, and not working now
##  print('Saved metric values')

if __name__ == "__main__":
    main()


