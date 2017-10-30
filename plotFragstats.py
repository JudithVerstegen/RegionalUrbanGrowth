import pickle
from collections import deque
import os
import numpy as np
##import parameters
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab

# swap function from Derek
def swapXandYInArray(a):
  b=np.reshape(a,a.size,order='F').reshape((a.shape[1],a.shape[0])) 
  return b

def plot(listOfStats, filterSteps, start, validation=False):

  t = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],\
       [15],[16],[17],[18],[19],[20],[21],[22]]
  t2 = np.array(range(1,23,1))
  t_obs = [[10],[16],[22]]
  plt.gcf().clear()
  plt.figure(1, figsize=(1,1))

  #plt.rcParams['axes.color_cycle'] = ['k']
  font = {'fontname':'Tahoma','fontsize':9}
  x=1
  for aStat in listOfStats:  
    allMeans = []
    upper = []
    lower = []
    allObs = []
    if validation == True:
      # NOT IMPLEMENTED FOR URBAN
      pass
    else:
      fileName = os.path.join('results', aStat + '.npy')
      obsName = os.path.join('results', aStat + '_obs.npy')

    # compute mean and 95% conf interval
    array = np.load(fileName)
    # squeezing necessary to remove extra dimension
    # otherwise plt.plot does not work
    array = np.squeeze(array)
    allMeans = np.median(array, axis = 1)
    lower = np.percentile(array, 2.5, axis = 1)
    upper = np.percentile(array, 97.5, axis = 1)
##    print lower
##    print upper

    # observations
    obs_array = np.load(obsName)
    obs_array = np.squeeze(obs_array)
    obs_allMeans = np.median(obs_array, axis = 1)
    obs_lower = np.percentile(obs_array, 2.5, axis = 1)
    obs_upper = np.percentile(obs_array, 97.5, axis = 1)
##    print obs_lower
##    print obs_upper


    for aLine in range(0,len(allMeans[0,:])):
      current = plt.subplot(3, 3, x)
      plt.plot(t, allMeans[:,aLine], 'k')
      current.fill_between(t2, lower[:,aLine], upper[:,aLine], \
                           facecolor='grey', alpha=0.5, label="95% confidence")
      ##plt.plot(t_obs, obs_allMeans[:,aLine], 'b+', ms=2.0)
      plt.errorbar(t_obs, obs_allMeans[:,aLine], \
                   yerr=[obs_lower[:,aLine], obs_upper[:,aLine]], 
                  fmt='o', ecolor='blue', capthick=2)

      current.set_xlim(0,len(t)+1)
      current.set_xticklabels(range(start, start + len(t2), 5))
      # create lines at the filter time steps
      ymin, ymax = current.get_ylim()
      bins = np.arange(0, ymax, 20)
      if ymax > 500:
        current.set_ylim(0,500)
      ymin, ymax = current.get_ylim()
      bins = np.arange(0, 600, 100)
      current.set_yticks(bins)
      
      for aMoment in filterSteps:
        current.vlines(aMoment, ymin, ymax, color='k', linestyles='dashed')
      x+=1
    plt.setp(current.get_xticklabels() + current.get_yticklabels(), fontsize=12)
   

  F = pylab.gcf()
  F.set_size_inches((15, 8))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                      wspace=0.2, hspace=0.3)
  if validation == True:
    plt.savefig('fragstats_val.png', dpi=300, format='png', orientation='landscape')  
  else:
    plt.savefig('fragstats.png', dpi=300, format='png', orientation='landscape')    

### test
start = 1990
plot(['nr'], [], start)
