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
obsSampleNumbers = [1] #range(1,20+1,1) <- for stochastic model
obsTimeSteps = [10,16,22,28]

# Path to the folder with the metrics stored
country = parameters.getCountryName()
resultFolder = os.path.join(os.getcwd(),'results',country)

#################
### FUNCTIONS ###
#################

def openPickledSamplesAndTimestepsAsNumpyArray(basename,samples,timesteps, \
                                               obs=False):
  output=[]
  
  for timestep in timesteps:
    allSamples=[]
    
    for sample in samples:
      # Read in the parameters
      pName = 'parameters_' + str(sample) + '.obj'
      pFileName = os.path.join(resultFolder, str(sample), pName)
      filehandler = open(pFileName, 'rb') 
      pData = pickle.load(filehandler)
      pArray = np.array(pData, ndmin=1)
    
      # If we are working with the observed data (CLC data):
      if obs:
        name = generateNameT(basename, timestep)
        fileName = os.path.join('observations', country, 'realizations', \
                                str(sample), name)
        data = metrics.map2Array(fileName, os.path.join('input_data', country, 'sampPoint.col'))

      # If we are working with the observed values:  
      else:
        theName = basename + str(timestep) + '.obj'
        fileName = os.path.join(resultFolder, str(sample), theName)
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
      allSamples.append([pArray,array]) # test if this would work??????????????
    output.append(allSamples)

  outputAsArray=np.array(output)
  return outputAsArray

def saveSamplesAndTimestepsAsNumpyArray(basename, samples, timesteps, \
                                        obs=False):
  output = openPickledSamplesAndTimestepsAsNumpyArray(basename, samples,\
                                                      timesteps, obs)
  if obs:
    fileName = os.path.join("results", country, basename + '_obs')
  else:
    fileName = os.path.join("results", country, basename)
  np.save(fileName, output)

#################################
### SAVE OUTPUTS OF THE MODEL ###
#################################
  
# now save all outputs of the model as one array per variable
# so that we can delete all number folders
 
variables = parameters.getSumStats()
print("Save modelled and observed metrics: ", variables)

# For the modelled metrics:
for aVariable in variables:
  saveSamplesAndTimestepsAsNumpyArray(aVariable, sampleNumbers, \
                                      timeSteps)

# For the observed values:
for aVariable in variables:
  saveSamplesAndTimestepsAsNumpyArray(aVariable, obsSampleNumbers,obsTimeSteps, True)

np.load(os.path.join("results", country, 'fd.npy'))
'''
# TEST

output = openPickledSamplesAndTimestepsAsNumpyArray('np', sampleNumbers, \
                                                    timeSteps)
print('np:')
#print(output)
# Output is indexed as array[time, sample, x, y]
# So, all samples for time step 3 is output[2,:,:,0]
# Last zero can also be :.

print(output[:,:,-1,:,0])
'''

