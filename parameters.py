"""Land use change model, designed for Brazil
Judith Verstegen 2012-01-03

"""

def getNrTimesteps():
  """Return nr of time steps.

  e.g. 2005 to 2030 is 26 time steps."""

  timesteps = 2
  return timesteps

def getNrSamples():
  """Return nr of Monte Carlo samples required.

  If Monte Carlo isn't required fill in 1; no statistics will be calculated."""
  
  samples = 2
  return samples

def getCountryName():
  """ Returns the case study symbol """

  name = 'PL'
  return name
  
  
def getCovarOn():
  """Return 1 if filtering with covariance matrix is required."""  
  on = 1
  return on

def getSumStats():
  # 'np': Number of patches
  # 'mp': Mean patch size
  # 'pd': Patch density
  # 'shdi': Shannon's diveristy index <- not implemented
  # 'cilp': Compactness index of the largest patch <- not implemented
  # 'awmpfd': Area weighted mean patch fractal dimension <- not implemented
  # 'fd': Fractal dimension
  
  sumStats = ['np', 'fd', 'mp', 'np', 'pd']
  return sumStats

def getCovarName():
  name = 'cov_nrz'
  return name

def getConversionUnit():
  """Return conversion unit for max yield unit to square meters.

  e.g. when max yield in ton/ha fill in 10000."""

  toMeters = 10000
  return toMeters

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
  suitFactorDict[1] = [1, 6, 4, 5]
  return suitFactorDict

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
  variableDict1[1] = [500, 0.7]
  variableDict1[2] = [0.3]
  variableDict1[4] = {4:1, 99:0.5}
  variableDict1[6] = [1]
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
  
  
