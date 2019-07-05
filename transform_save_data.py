# -*- coding: cp1252 -*-
import pickle
#from collections import deque
import os
import metrics
import numpy as np
import parameters
from pcraster.framework import *

#### Script to read in the metrics saved as the result of the LU_urb.py script.
#### Metrics are transformed into an array

# Get the number of samples and number of time step defined in the parameter.py script
nrOfSamples=parameters.getNrSamples()
nrOfTimesteps=parameters.getNrTimesteps()

sampleNumbers=range(1,nrOfSamples+1,1)
timeSteps=range(1,nrOfTimesteps+1,1)

# Get the observed time steps. Time steps relate to the year of the CLC data, where 1990 was time step 0.
#obsSampleNumbers=range(1,20+1,1)
obsTimeSteps = [10,16,22,28]


def openPickledSamplesAndTimestepsAsNumpyArray(basename,samples,timesteps, \
                                               obs=False):
  t=1
  output=[]
  for timestep in timesteps:
##    print 'timestep ' + str(timestep) + ' done,',
    allSamples=[]
    for sample in samples:
      if obs:
        name = generateNameT(basename, timestep)
        fileName = os.path.join('observations', 'realizations', \
                                str(sample), name)
        data = metrics.map2Array(fileName, 'input_data/sampPoint.col')
      else:
        fileName = os.path.join(str(sample), basename + str(timestep) + '.obj')
        filehandler = open(fileName, 'rb') 
        data=pickle.load(filehandler)
        # if it is a dictionary, get the sugar cane parameters (lu type 6)
        if type(data) == dict:
          data = data.get(1)
        filehandler.close()
      # if the loaded data was not yet an array, make it into one
      # minimum number of dimensions is one, to prevent a zero-dimension array
      # (not an array at all)
      array = np.array(data, ndmin=1)
      # add an extra dimension that would normally be y, if the data was a map
      # so that Derek's plot functions can be used
      array = array.reshape(len(array),1)
      allSamples.append(array)
    output.append(allSamples)
  outputAsArray=np.array(output)
  return outputAsArray

def saveSamplesAndTimestepsAsNumpyArray(basename, samples, timesteps, \
                                        obs=False):
  output = openPickledSamplesAndTimestepsAsNumpyArray(basename, samples,\
                                                      timesteps, obs)
  if obs:
    fileName = os.path.join("results", basename + '_obs')
  else:
    fileName = os.path.join("results", basename)
  np.save(fileName, output)

##########

# now save all outputs of the model as one array per variable
# so that we can delete all number folders
variables = ['nr']
for aVariable in variables:
  saveSamplesAndTimestepsAsNumpyArray(aVariable, sampleNumbers, \
                                      timeSteps)
saveSamplesAndTimestepsAsNumpyArray('nr', obsSampleNumbers, \
                                    obsTimeSteps, True)
#test
  
##output = openPickledSamplesAndTimestepsAsNumpyArray('weights', sampleNumbers, \
##                                                    timeSteps)
##print output
### Output is indexed as array[time, sample, x, y]
### So, all samples for time step 3 is output[2,:,:,0]
### Last zero can also be :.
##print output[:,-1,:,0]
