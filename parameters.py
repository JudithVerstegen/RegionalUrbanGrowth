"""Urban expansion model, parameters
Judith Verstegen 2019-06-07
"""

def getWorkDir():
  import os
  return os.getcwd()#'/scratch/tmp/k_goch01'

def getNrTimesteps():
  """Return nr of time steps.

  e.g. 2005 to 2030 is 26 time steps.
  In the model 1990 - 2018 CLC data are used, including starting and ending date, thus 29 time steps.
  Time step nr 1 is 1990"""

  timesteps = 29
  return timesteps

def getObsTimesteps():
  """Return the time steps used for calibration and validation.

  e.g. 2000 in time step nr 11.
  In the model 2000 and 2006 data are used for calibration, and 2012 and 2018 are used for validation."""
  
  obsTimeSteps = [1,11,17,23,29]
  return obsTimeSteps

def getObsYears():
  """Returns a dictionairy containing years corresponding to the observation time steps."""
  return {1:1990,11:2000,17:2006,23:2012,29:2018}

def getCalibrationPeriod():
  """Returns the indexes of years for calibration and validation from getObsYears().values():
  e.g. [1990, 2000, 2006, 2012, 2018]
  Scenario 1: calibration period 2000 - 2006, validation period 2012 - 2018
  Scenario 2: calibration period 2012 - 2018, validation period 2000 - 2006"""

  period = {
    1: {
      'calibration': [1,2],
      'validation': [3,4]
      },
    2: {
      'calibration': [3,4],
      'validation': [1,2]
      }}
  return period

def getNumberOfZones():
  """ Returns the number of zones, in which the case study area will be divided.
      Some metrics will be calculated for each zone seperately. """
  
  numberofZones = 16
  return numberofZones

def getNrSamples():
  """Return nr of Monte Carlo samples required.

  If Monte Carlo isn't required fill in 1; no statistics will be calculated."""
  
  samples = 1 #100
  return samples

def getParametersforCalibration():
  """Return min, max and step of the parameter to be used in the calibration
     minParameter needs to be >= 0
     maxParameter needs to be <= 1
    [minParameter, maxParameter, stepSize] """
  return [0.0, 1.0, 0.1]

def getCountryName():
  """ Returns the case study symbol """
  # case studies: 'IT', 'IE', 'PL'

  name = 'PL'
  return name

def getCaseStudies():
  """ Returns the case studies for analysis """
  # case studies: 'IT', 'IE', 'PL'

  cases = ['IE', 'IT', 'PL']
  return cases
  
def getCovarOn():
  """Return 1 if filtering with covariance matrix is required."""  
  on = 1
  return on

def getSumStats():
  # 'np': Number of patches
  # 'mp': Mean patch size
  # 'cilp': Compactness index of the largest patch
  # 'fdi': Fractal dimension index
  # 'wfdi': Area weighted mean patch fractal dimension index
  # 'pd': Patch density
  # 'cohes': # Patch Cohesion Index in a zone
  # 'ed': # Edge Density
  # 'lpi': # Largest Patch Index
  # 'contag': # Contagion Index 
  
  sumStats = ['wfdi','cohes'] #['cilp', 'fdi', 'wfdi', 'pd', 'cohes', 'ed', 'lpi', 'contag']
  return sumStats

def getLocationalAccuracyMetric():
  """ Returns locational metric used in the calibration """
  # Locational metrics: 'K', 'Ks', 'A'

  locationalMetric = ['A']
  return locationalMetric

def getCovarName():
  name = 'cov_nrz'
  return name

def getConversionUnit():
  """Return conversion unit for max yield unit to square meters.

  e.g. when max yield in ton/ha fill in 10000."""

  toMeters = 10000
  return toMeters

def getAlphaValue():
  """alpha is a scalable parameter that controls the stochastic effect,
  with −10 corresponding to almost no randomness and 1 to high randomness.
  A value of 0.5 is usually acceptable for a fairly weak random effect.
  Values between 0 and 1 are usually appropriate.

  e.g. alpha = 0.6 was applied to reflect the urban sprawl effect in
  (Barredo, Demicheli, Lavalle, Kasanko, & McCormick, 2004)"""
  
  alpha = 0.6
  return alpha

def getLandUseList():
  """Return list of landuse types in ORDER of 'who gets to choose first'."""
  landUseList = [1]
  return landUseList

def getRelatedTypeDict():
  """Return dictionary which type (key) is related to which others (items).

  e.g. relatedTypeDict[3] = [1, 2, 3, 7] means:
  land use type 3 is related to types 1, 2, 3 and 7.
  This is used in suitability factor 1 about neighbors
  of the same or a related type."""
  
  relatedTypeDict = {}
  relatedTypeDict[1] = [1]
  return relatedTypeDict

def getSuitFactorDict():
  """Return dictionary which type (key) has which suit factors (items).

  e.g. suitFactorDict[1] = [1, 2, 4, 5, 6, 9] means:
  land use type 1 uses suitability factors 1, 2, 4, 5, 6 and 9."""
  
  suitFactorDict = {}
  suitFactorDict[1] = [1, 2, 3, 4]
  return suitFactorDict

def getNumberofIterations():
  """ Returns number of iterations depnded on the number, min, max and step of the parameters"""
  import numpy as np
  suma = 0
  count = 0
  min_p = getParametersforCalibration()[0]
  max_p = getParametersforCalibration()[1]
  stepsize = getParametersforCalibration()[2]
  param_steps = np.arange(min_p, max_p + 0.1, stepsize)
  for step in range(0,len(param_steps)):
    # Round parameters to avoid issues with the precision
    param_steps[step] = round(param_steps[step],4)
  for p1 in param_steps:
      for p2 in param_steps:
          for p3 in param_steps:
              for p4 in param_steps:
                  suma = p1+p2+p3+p4
                  if (suma>0.9999 and suma < 1.0001):
                      count = count + 1
  return count

def getWeightDict():
  """Return dictionary how a type (key) weights (items) its suit factors.

  e.g. weightDict[1] = [0.3, 0.1, 0.2, 0.1, 0.2, 0.1] means:
  land use type 1 has suitability factor - weight:
  1 - 0.3
  2 - 0.1
  4 - 0.2
  5 - 0.1
  6 - 0.2
  9 - 0.1

  Note that the number and order of weights has to correspond to the
  suitbility factors in the previous method."""
  
  weightDict = {}
  ## A list with weights in the same order as the suit factors above
  weightDict[1] = [0.3, 0.3, 0.2, 0.2]

  return weightDict

def getVariableSuperDict():
  """Return nested dictionary for which type (key1) which factor (item1
  and key2) uses which parameters (items2; a list).

  e.g. variableDict1[2] = [-1, 10000, 1, 2] means:
  land use type 1 uses in suitability factor 2 the parameters:
  -1 for direction of the relation (decreasing)
  10000 for the maximum distance of influence
  1 for friction
  and relation type 'inversely proportional' (=2).

  An explanation of which parameters are required for which suitability
  factor is given in the manual of the model."""

  variableSuperDict = {}
  variableDict1 = {}
  variableDict1[1] = [1000, 0.7]
  variableDict1[2] = [0.5]
  variableDict1[3] = [0.5]
  variableDict1[4] = {1:0, 2:0, 3:0.5, 4:1} 
  variableSuperDict[1] = variableDict1
  return variableSuperDict

def getNoGoLanduseTypes():
  """Return a list of land use type numbers that cannot be changed

  At the moment this only works for static land uses
  The noGo map is calculated once at the beginning of the run."""

  noGoLanduse = [2]
  return noGoLanduse

def getYieldMapName(typeNr):
  """Return the name of the yield map for this land use type (mandatory)."""
  yieldMapNameDict = {}
  yieldMapNameDict[1] = 'input_data/onemask'

  needed = yieldMapNameDict.get(typeNr)
  return needed

def getColFiles():
  """Return a dictionairy of metrics and corresponding col files

  For each meric a value is saved for one or more cells.
  The cell coordinates are created in create_initial_maps.py and saved in .col files"""
  
  colFiles = {
    'fdi': 'sampPoint.col',         # each zone
    'wfdi': 'sampPoint.col',        # each zone
    'cilp': 'sampSinglePoint.col',  # single point in the middle of the study area
    'pd': 'sampSinglePoint.col',    # single point in the middle of the study area
    'urb': 'sampPointNr.col',       # point for each cell
    'cohes': 'sampPoint.col',       # each zone
    'ed': 'sampPoint.col',          # each zone
    'lpi': 'sampSinglePoint.col',   # single point in the middle of the study area
    'contag': 'sampPoint.col'    # each zone
    }
  return colFiles
  
  
  
