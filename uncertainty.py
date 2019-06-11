"""Land use change model, designed for Ukraine
Judith Verstegen, 2011-10-12

"""
import numpy as np
import parameters
from pcraster import *
from pcraster.framework import *
import math
import os

inputfolder = os.path.join('input_data', parameters.getCountryName())
nullMask = readmap(inputfolder + '/nullmask')

def getLogRealization(mu, sigma):
  realization = np.random.lognormal(mu, sigma)
  return realization

def getYieldMap(deterYieldMap):
  sd = 0.3
  stochMap = deterYieldMap + normal(1) * sd * deterYieldMap
  stochMap = min(stochMap, 1)
  stochYieldMap = max(stochMap, 0)
  return stochYieldMap

def getInitialLandUseMap(otherLanduses):
  randomNumber = random.randint(1,50)
  path = os.path.join('initial', 'sc_' + str(randomNumber) + '.map')
  print('initial land use map is', path)
  caneMap = readmap(path)
  onlyCane = ifthen(caneMap == 1, nominal(6))
  newLanduse = cover(onlyCane, otherLanduses)
  return newLanduse

def getWeights1(suitFactorsPerLUType):
  """Return a dictionary with the weights of the factors per LU type."""
  weightDict = {}
  nrOfTypes = len(suitFactorsPerLUType)
  for aType in suitFactorsPerLUType.keys():
    nrWeights = len(suitFactorsPerLUType.get(aType))
    weights = [0.0] * nrWeights
    for aWeight in range(0, nrWeights):
      weights[aWeight] = np.random.random()
    weightDict[aType] = weights
  for aType in suitFactorsPerLUType.keys():
    nrWeights = len(suitFactorsPerLUType.get(aType))
    normalized = [0.0] * nrWeights
    unNormalized = weightDict.get(aType)
    for aWeight in range(0, nrWeights):
      normalized[aWeight] = unNormalized[aWeight]/sum(unNormalized)
    weightDict[aType] = normalized
    if sum(normalized) != 1.0: print('ERROR', sum(normalized))
  print(weightDict)
  return weightDict

def getWeights2(suitFactorsPerLUType):
  """Return a dictionary with the weights of the factors per LU type."""
  weightDict = {}
  nrOfTypes = len(suitFactorsPerLUType)
##  print 'number lu types is', nrOfTypes
  for aType in suitFactorsPerLUType.keys():
    print(aType)
    nrWeights = len(suitFactorsPerLUType.get(aType))
    print(nrWeights)
    uninformed = 1
##    while (slope_weight < 0.4) or (neigh_weight > 0.3) or \
##      (dist_weight > 0.4) or (yield_weight > 0.4):
    while uninformed == 1:
      weights = [0.0] * nrWeights
      for aWeight in range(0, nrWeights):
          sumWeights = float(sum(weights))
          # If it's the final weight
          if aWeight == (nrWeights - 1):
              weight = 1.0 - sumWeights
##              print sumWeights, 'situation final', weight
          # Most often occuring situation
          elif sumWeights < 1.0:
              weight = np.random.uniform(0.0, (1.0 - sumWeights))
##              print 'situation 1', weight
          # If the sum is already one (should not be possible;
          # only through rounding)
          else:
              weight = 0.0
##              print 'situation 2', weight
          weights[aWeight] = weight
      np.random.shuffle(weights)
      uninformed = 0
      
    sumWeights = float(sum(weights))
##    print sumWeights
##    if sum(weights) != 1.0: print 'ERROR', sum(weights)
    weightDict[aType] = weights
  print(weightDict)
  return weightDict

def getDem(deterDem):
  """Return a stochastic dem with standard deviation as defined here."""
  # error at 90% conf is 5.5 m for SA (Farr et al)--> sd = 6.2 / 1.64 = 3.96 m
  sd = 3.96
  stochDem = deterDem + normal(1) * sd
  return stochDem

def getSuitabilityParameters(suitFactorsPerLUType):
  """Return 1 when the max distance should have a random error.

  When 1 the maximum distance for the suitability factors 2, 3 and 4
  varies uniformly between 1 celllength and 2 * max distance,
  e.g. with cellsize of 1 km2 and max distance 5000
  max distance varies between 1000 and 10000 m"""
  variableSuperDict = {}
  for aType in suitFactorsPerLUType.keys():
    print(aType)
    variableDict = {}
    factors = suitFactorsPerLUType.get(aType)
    print(factors)
    for aFactor in factors:
      # factors that need an 'a' for y=x**a
      if aFactor in [2,3]:
        # see test script for mu and sigma explanation
        sigma = 1.8
        mu = 0 #sigma ** 2, was wrong before, with this old meth modus is 1
        a = getLogRealization(mu, sigma)
        print(aFactor, 'a is', a)
        variableDict[aFactor] = [a]
      elif aFactor == 1:
        # neighborhood suitability that needs window length
        # and x value of parabole top
        sigma = 0.7
        mu = math.log(3000) + sigma**2
        windowLength = 20000
##        while windowLength > 10000:
        windowLength = round(getLogRealization(mu, sigma), 1)
        top = 1
        # added to limit the range op top to [0, 0.5]
##        while (top > 0.6) or (top < 0.3):
        top = round(float(mapuniform()),2)
        #top = round(float(mapuniform()/2),2)
        variableDict[aFactor] = [windowLength, top]
        print('window', round(windowLength, 1), 'top', round(top, 2))
      elif aFactor == 5:
        pass
      elif aFactor == 4:
        # elasticity suitability
        sigma = 0.2
        realization4 = round(float(mapnormal()),2)
        print(realization4)
        elasticity4 = 0.5 + realization4 * sigma
        variableDict[aFactor] = {4:elasticity4}
      else:
        print('PROBLEMO suit factor number')
      variableSuperDict[aType] = variableDict
  return variableSuperDict

