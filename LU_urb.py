"""Urban growth model
Judith Verstegen, 2019-05-09

"""
import random
random.seed(12)
import math
from pcraster import *
from pcraster.framework import *
setrandomseed(10)
import parameters
import uncertainty
import pickle 
import metrics
import numpy as np
np.random.seed(10)
import mcaveragevariance

#######################################

class LandUseType:
  def __init__(self, typeNr, environment, relatedTypeList, suitFactorList, \
               weightList, variableDict, noise, nullMask, \
               windowLengthRealization):
    """Create LandUseType object that represents a class on the land use map.

    Takes ten arguments:
    typeNr -- class nr of the land use type on the land use map
    environment -- global land use map that will evolve
    relatedTypeList -- list with land use type next to which growth is preferred
    suitFactorList -- list of suitability factors the type takes into account
    weightList -- list of relative weights for those factors
    variableDict -- dictionary in which inputs for factors are found
    noise -- very small random noise to ensure cells can't get same suitability
    nullMask -- map with value 0 for study area and No Data outside
    windowLengthRealization -- window length for neighborhood function (stoch)
    
    """
    
    self.typeNr = typeNr
    self.environment = environment
    self.relatedTypeList = relatedTypeList
    self.suitFactorList = suitFactorList
    self.weightList = weightList
    self.variableDict = variableDict
    self.nullMask = nullMask

    self.noise = noise
    self.toMeters = parameters.getConversionUnit()
    self.windowLengthRealization = windowLengthRealization
    
  def setEnvironment(self, environment):
    """Update the environment (land use map)."""
    self.environment = environment

  def createInitialMask(self, globalMapNoGo):
    """Now just the global no-go map."""
    self.mask = globalMapNoGo

  def normalizeMap(self, aMap):
    """Return a normalized version of the input map."""
    mapMax = mapmaximum(aMap)
    mapMin = mapminimum(aMap)
    diff = float(mapMax - mapMin)
    if abs(diff) < 0.000001:
      normalizedMap = (aMap - mapMin) / 0.000001
    else:
      normalizedMap = (aMap - mapMin) / diff
    return normalizedMap
  
  ## 1
  def getNeighborSuitability(self):
    """Return suitability map based on nr of neighors with a related type."""
    booleanSelf = pcreq(self.environment, self.typeNr)
    for aType in self.relatedTypeList:
      booleanMap = pcreq(self.environment, aType)
      booleanSelf = pcror(booleanSelf, booleanMap)
    scalarSelf = scalar(booleanSelf)
    # Count nr of neighbors with 'true' in a window with length from parameters
    # and assign this value to the centre cell
    variableList = self.variableDict.get(1)
    windowLength = variableList[0]
    nrNeighborsSameLU = windowtotal(scalarSelf, windowLength) - scalarSelf
    # The nr of neighbors are turned into suitability values between 0 and 1
    maxNr = windowtotal(self.nullMask + 1, windowLength) - 1
  
    # NEW
    # f [0,1]
    f = variableList[1]
    neighborSuitability = -1*(nrNeighborsSameLU**2) + f * 2 * maxNr *\
                               nrNeighborsSameLU
    neighborSuitability = self.normalizeMap(neighborSuitability)
    report(neighborSuitability, 'suit_neigh' + str(self.typeNr))
##    maxNr = ((windowLength / celllength())**2) - 1
##    report(maxNr, 'test_old')
##    neighborSuitability = nrNeighborsSameLU / maxNr
##    report(neighborSuitability, 'neighborSuitability_old')
    return neighborSuitability

  ## 2
  def getDistanceSuitability(self, spreadMap):
    """Return suitability map based on distance to roads."""
    variableList = self.variableDict.get(2)
    a = variableList[0]
    normalized = self.normalizeMap(spreadMap)
    roadSuitability = 1 - (normalized ** a)  
    report(roadSuitability, 'suit_station' + str(self.typeNr))
    return roadSuitability

  ## 3
  def getTravelTimeCityBorder(self):
    """Return suitability map based on distance to large cities."""
    booleanSelf = pcreq(self.environment, self.typeNr)
    clumps = clump(ifthen(booleanSelf == 1, boolean(1)))
    sizes = areaarea(clumps)
    city = cover(sizes == mapmaximum(sizes), boolean(0))

    dist = spread(city, 0, self.friction)
    variableList = self.variableDict.get(3)
    a = variableList[0]
    normalized = self.normalizeMap(dist)
    travelSuitability = 1 - (normalized ** a)  
    report(travelSuitability, 'suit_travel' + str(self.typeNr))
    return travelSuitability

  ## 4
  def getCurrentLandUseSuitability(self):
    """Return suitability map based on current land use type."""
    variableDict = self.variableDict.get(4)
    current = self.nullMask
    for aKey in variableDict.keys():
      current = ifthenelse(pcreq(self.environment, aKey), \
                           variableDict.get(aKey), current)
    currentLandUseSuitbaility = self.normalizeMap(current)
    report(currentLandUseSuitbaility, 'suit_curLu' + str(self.typeNr))
    return currentLandUseSuitbaility

 
  def createInitialSuitabilityMap(self, distmap, yieldFrac, friction):
    """Return the initial suitability map, i.e. for static factors.

    Given the maps:
    distmap -- distances to something 
    yieldFrac -- fraction of maximum yield that can be reached in a cell
                (is kept for potential later use of population density)
    friction -- friction computed from road map (for distance urban border)
    
    Uses a lists and two dictionaries created at construction of the object:
    factors -- the names (nrs) of the suitability factors (methods) needed
    parameters -- the input parameters for those factors
    weights -- the weights that belong to those factors (how they're combined).

    """

    self.weightInitialSuitabilityMap = 0
    self.initialSuitabilityMap = spatial(scalar(0))
    self.yieldFrac = yieldFrac
    self.friction = friction
    i = 0
    # For every number in the suitability factor list
    # that belongs to a STATIC factor
    # the corresponding function is called providing the necessary parameters
    # and the partial suitability map is added to the total
    # taking into account its relative importance (weight)
    for aFactor in self.suitFactorList:
      if aFactor == 2:
        self.initialSuitabilityMap += self.weightList[i] * \
                                 self.getDistanceSuitability(distmap)
        self.weightInitialSuitabilityMap += self.weightList[i]      
      elif aFactor == 4:
        self.initialSuitabilityMap += self.weightList[i] * \
                          self.getCurrentLandUseSuitability()
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor in (1, 3):
        ## Dynamic factors are captured in the total suitability map
        pass
      else:
        print('ERROR: unknown suitability factor for landuse', self.typeNr)
      i += 1
    print('weight of initial factors of', self.typeNr, \
          'is', self.weightInitialSuitabilityMap)
    self.initialSuitabilityMap += self.noise
##    report(self.initialSuitabilityMap, 'iniSuit' + str(self.typeNr))

  def getTotalSuitabilityMap(self):
    """Return the total suitability map for the land use type.

    Uses a lists and two dictionaries:
    factors -- the names (nrs) of the suitability factors (methods) needed
    parameters -- the input parameters for those factors
    weights -- the weights that belong to those factors (how they're combined).

    """

    suitabilityMap = spatial(scalar(0))
    i = 0
    # For every number in the suitability factor list
    # that belongs to a DYNAMIC factor
    # the corresponding function is called providing the necessary parameters
    # and the partial suitability map is added to the total
    # taking into account its relative importance (weight)
    for aFactor in self.suitFactorList:
      if aFactor == 1:
        suitabilityMap += self.weightList[i] * self.getNeighborSuitability()
      elif aFactor == 3:
        suitabilityMap += self.weightList[i] * \
                                      self.getTravelTimeCityBorder()
      elif aFactor in (2, 4):
        # Static factors already captured in the initial suitability map
        pass
      else:
        print('ERROR: unknown suitability factor for landuse', self.typeNr)
      i += 1
    suitabilityMap += self.weightInitialSuitabilityMap * \
                      self.initialSuitabilityMap
    self.totalSuitabilityMap = self.normalizeMap(suitabilityMap)
    report(self.totalSuitabilityMap, 'suit_tot' + str(self.typeNr))
    return self.totalSuitabilityMap

  def setMaxYield(self, maxYield):
    """Set the maximum yield in this time step using the input from the tss."""
    convertedMaxYield = (maxYield / self.toMeters) * cellarea()
    # max yield on cells currently occupied
    ownMaxYield = ifthen(self.environment == self.typeNr, convertedMaxYield)
    # maximum yield PER CELL
    self.maxYield = float(mapmaximum(ownMaxYield))
##    report(ownMaxYield, 'test')
    # potential yield on all cells
    self.yieldMap = self.yieldFrac * self.maxYield
    
  def updateYield(self, env):
    """Calculate total yield generated by cells occupied by this land use."""
    # Current cells taken by this land use type
    self.currentYield = ifthen(env == self.typeNr, self.yieldMap)
    self.totalYield = float(maptotal(self.currentYield))

  def allocate(self, demand, tempEnvironment, immutables):
    """ Assess total yield, compare with demand and add or remove difference."""
    self.setEnvironment(tempEnvironment)
    self.updateYield(tempEnvironment)
##    report(self.currentYield, 'currentYield' + str(self.typeNr))
    ownDemand = ifthen(self.environment == self.typeNr, demand)
    self.demand = float(mapmaximum(ownDemand))
    if self.demand < 0.0:
      self.demand = float(0.0)
    print('demand is:', self.demand)
    print('current total is:', self.totalYield)
    if self.totalYield > self.demand:
      print('remove')
      self.remove()
    elif self.totalYield < self.demand:
      print('add')
      self.add(immutables)
    else:
      print('do nothing')
    newImmutables = ifthenelse(self.environment == self.typeNr, boolean(1),\
                               immutables)
    return self.environment, newImmutables
    
  def add(self, immutables):
    """Add cells of this land use type until demand is fullfilled."""
    # Remove cells from immutables (already changed)
    self.totalSuitabilityMap = ifthen(pcrnot(pcror(immutables, self.mask)), \
                                      self.totalSuitabilityMap)
    # Remove cells already occupied by this land use
    self.totalSuitabilityMap = ifthen(self.environment != self.typeNr, \
                                      self.totalSuitabilityMap)
    # Determine maximum suitability and allocate new cells there
    mapMax = mapmaximum(self.totalSuitabilityMap)
    print('start mapMax =', float(mapMax))
    ordered = order(self.totalSuitabilityMap)
    maxIndex = int(mapmaximum(ordered))
    diff = float(self.demand - self.totalYield)
    x = int(maxIndex - diff / self.maxYield)
    xPrev = maxIndex
    i = 0
    tempEnv = self.environment
    while diff > 0 and xPrev > x:
      print('cells to add', int(maxIndex - x))
      if x < 0:
        print('No space left for land use', self.typeNr)
        break
      else:
        # The key: cells with maximum suitability are turned into THIS type
        tempEnvironment = ifthen(ordered > x, nominal(self.typeNr))
        tempEnv = cover(tempEnvironment, self.environment)

        # Check the yield of the land use type now that more land is occupied
        self.updateYield(tempEnv)
        i += 1
        xPrev = x
        # Number of cells to be allocated
        diff = float(self.demand - self.totalYield)
        x -= int(diff / self.maxYield)
    self.setEnvironment(tempEnv)
    print('iterations', i, 'end yield is', self.totalYield)

  def remove(self):
    """Remove cells of this land use type until demand is fullfilled."""
    # Only cells already occupied by this land use can be removed
    self.totalSuitabilityMap = ifthen(self.environment == self.typeNr, \
                                      self.totalSuitabilityMap)
    ordered = order(self.totalSuitabilityMap)
    mapMin = mapminimum(self.totalSuitabilityMap)
    print('start mapMin =', float(mapMin))
    diff = float(self.totalYield - self.demand)
    x = int(diff / (self.maxYield * 0.8))
    xPrev = 0
    i = 0
    tempEnv = self.environment
    while diff > 0 and xPrev < x and i < 100:
      print('cells to remove', x)
      # The key: cells with minimum suitability are turned into 'abandoned'
      tempEnvironment = ifthen(ordered < x, nominal(99))
      tempEnv = cover(tempEnvironment, self.environment)
      
      # Check the yield of the land use type now that less land is occupied
      self.updateYield(tempEnv)
      i += 1
      xPrev = x
      diff = float(self.totalYield - self.demand)
      if math.fmod(i, 40) == 0:
        print('NOT getting there...')
        # Number of cells to be allocated
        x = 2 * (x + int(diff / self.maxYield))      
      else:
        # Number of cells to be allocated
        x += int(diff / self.maxYield)
    self.setEnvironment(tempEnv)
    print('iterations', i, 'end yield is', self.totalYield)
##    report(self.environment, 'newEnv' + str(self.typeNr))


#######################################

class LandUse:
  def __init__(self, types, nullMask):
    """Construct a land use object with a nr of types and an environment."""
    self.types = types
    self.nrOfTypes = len(types)
    print('\nnr of dynamic land use types is:', self.nrOfTypes)
##    self.environment = environment
    # Map with 0 in study area and No Data outside, used for cover() functions
    self.nullMask = nullMask
    self.toMeters = parameters.getConversionUnit()

  def setInitialEnvironment(self, environment):
    """Update environment of the 'overall' class ONLY."""
    self.environment = environment    

  def setEnvironment(self, environment):
    """Update environment of the 'overall' class and separate land use types."""
    self.environment = environment
    for aType in self.landUseTypes:
      aType.setEnvironment(self.environment)
    
  def createLandUseTypeObjects(self, relatedTypeDict, suitabilityDict, \
                               weightDict, variableSuperDict, noise):
    """Generate an object for every dynamic land use type.

    Make objects with:
    typeNr -- class nr in land use map
    environment -- global land use map
    relatedTypes -- list with land use types next to which growth is preferred
    suitFactors -- list with nrs of the needed suitability factors
    weights -- list with relative weights for those factors
    variables -- dictionary with inputs for those factors
    noise -- small random noise that determines order when same suitability

    """
    # List with the land use type OBJECTS
    self.landUseTypes = []
    windowLengthRealization = float(mapnormal())
    
    for aType in self.types:      
      # Get the list that states witch types the current types relates to
      relatedTypeList = relatedTypeDict.get(aType)
      # Get the right list of suitability factors out of the dictionary
      suitabilityList = suitabilityDict.get(aType)
      # Get the weights and variables out of the weight dictionary
      weightList = weightDict.get(aType)
      variableDict = variableSuperDict.get(aType)
      # Parameter list is notincluded yet
      self.landUseTypes.append(LandUseType(aType, self.environment, \
                                           relatedTypeList, suitabilityList, \
                                           weightList, variableDict, noise, \
                                           self.nullMask, \
                                           windowLengthRealization))
      
  def determineNoGoAreas(self, noGoMap, noGoLanduseList):
    """Create global no-go map, pass it to the types that add own no-go areas."""
    self.excluded = noGoMap
    # Check the list with immutable land uses
    if noGoLanduseList is not None:
      for aNumber in noGoLanduseList:
        booleanNoGo = pcreq(self.environment, aNumber)
        self.excluded = pcror(self.excluded, booleanNoGo)
    ##report(scalar(self.excluded), 'excluded')
    i = 0
    for aType in self.types:
      self.landUseTypes[i].createInitialMask(self.excluded)
      i += 1

  def determineDistanceToStations(self, mapStations):
    """Create map with distance to roads, given a boolean map with roads."""
    # stations now as boolean
    stations = pcrne(mapStations, 0)
    self.distStations = spread(stations, 0, 1)
    report(self.distStations, 'distStations.map')
    
  def loadDistanceMaps(self):
    """load the distance maps, when they cannot be kept in memory (fork)"""
##    print os.getcwd()
    self.distStations = readmap('distStations')
    self.relativeFriction = readmap('relativeFriction')

  def determineSpeedRoads(self, nominalMapRoads):
    """Create map with relative speed on raods, using boolean map with roads."""
    # By using the part below one can make a map of relative time to
    # reach a hub, giving roads a lower friction
    speed = cover(lookupscalar('speed.txt', nominalMapRoads), \
                  self.nullMask + 5)
    self.relativeFriction = 1.0/speed
    report(self.relativeFriction, 'relativeFriction.map')
  
  def calculateStaticSuitabilityMaps(self, stochYieldMap):
    """Get the part of the suitability maps that remains the same."""
    for aType in self.landUseTypes:
      # Check whether the type has static suitability factors
      # Those have to be calculated only once (in initial)
      aType.createInitialSuitabilityMap(self.distStations, stochYieldMap,\
                                        self.relativeFriction)

  def calculateSuitabilityMaps(self):      
    """Get the total suitability maps (static plus dynamic part)."""
    suitMaps = []
    for aType in self.landUseTypes:
      suitabilityMap = aType.getTotalSuitabilityMap()
      suitMaps.append(suitabilityMap)

  def allocate(self, maxYield, demand):
    """Allocate as much of a land use type as indicated in the demand tss."""
    tempEnvironment = self.environment
    immutables = self.excluded
    for aType in self.landUseTypes:
      aType.setMaxYield(maxYield)
      tempEnvironment, immutables = aType.allocate(demand, tempEnvironment, \
                                                   immutables)
    self.setEnvironment(tempEnvironment)    

  def getEnvironment(self):
    """Return the current land use map."""
    return self.environment

  def getSlopeMap(self):
    """Return the slope map."""
    return self.slopeMap
	
########################################################

class LandUseChangeModel(DynamicModel):
  def __init__(self, nr, weights):
    DynamicModel.__init__(self)
    # number for reference
    self.currentSampleNumber = nr
    # parameters to calibrate
    self.weightDict = {1: weights}
    # input and output folders
    country = parameters.getCountryName()
    output_mainfolder = os.path.join(os.getcwd(), 'results', country)
    if not os.path.isdir(output_mainfolder):
      os.mkdir(output_mainfolder)
    self.outputfolder = os.path.join(os.getcwd(), 'results', country, str(nr))
    if not os.path.isdir(self.outputfolder):
      os.mkdir(self.outputfolder)
    self.inputfolder = os.path.join('input_data', country)
    setclone(self.inputfolder + '/nullmask')
##    setglobaloption('nondiagonal')

    # Save the parameters as a list to the folder with the calculated metrics
    pName = 'parameters_' + str(nr) + '.obj'
    pPath = os.path.join(self.outputfolder, pName)
    parametersFile = open(pPath, 'wb')
    pickle.dump(weights, parametersFile)
    parametersFile.close()

  def initial(self):
    # create sample points
    self.nullMask = self.readmap(self.inputfolder + '/nullmask')
    self.oneMask = self.readmap(self.inputfolder + '/onemask')   
    # AT SOME POINT WITH STOCHASTIC INPUT
    # in that case land use should not include urban
    self.landuse = self.readmap(self.inputfolder + '/init_lu')
    self.initialUrb = self.landuse == 1
    self.roads = self.readmap(self.inputfolder + '/roads')
    self.noGoMap = cover(self.readmap(self.inputfolder + '/nogo'), \
                         boolean(self.nullMask))
    self.zones = readmap(self.inputfolder + '/zones')
    self.samplePoints = self.readmap(self.inputfolder + '/sampPoint')
    self.sumStats = parameters.getSumStats()
    self.yieldMap = scalar(self.oneMask)

    # List of landuse types in order of 'who gets to choose first'
    self.landUseList = parameters.getLandUseList()
    self.relatedTypeDict = parameters.getRelatedTypeDict()

    # Input values from parameters file
    self.suitFactorDict = parameters.getSuitFactorDict()
    self.variableSuperDict = parameters.getVariableSuperDict()
    self.noGoLanduseList = parameters.getNoGoLanduseTypes() 

    # Uniform map of very small numbers, used to avoid equal suitabilities
    self.noise = uniform(1)/10000
    
    # This part used to be the initial
    # Set seeds to be able to reproduce results
    random.seed(10)
    np.random.seed(10)
    setrandomseed(10)
    
    # Create the 'overall' landuse class
##    self.environment = uncertainty.getInitialLandUseMap(self.landuse)
    self.environment = self.landuse
    
    self.landUse = LandUse(self.landUseList, self.nullMask)
    self.landUse.setInitialEnvironment(self.environment)


    # Create an object for every landuse type in the list    
    self.landUse.createLandUseTypeObjects(self.relatedTypeDict, \
                                          self.suitFactorDict, \
                                          self.weightDict, \
                                          self.variableSuperDict, \
                                          self.noise)

    # Static suitability factors
    self.landUse.determineNoGoAreas(self.noGoMap, self.noGoLanduseList)
    self.landUse.loadDistanceMaps()
    self.landUse.calculateStaticSuitabilityMaps(self.yieldMap)

  def dynamic(self):
    timeStep = self.currentTimeStep()
    print('\ntime step', timeStep)
    
    # Get max yield and demand per land use type
    # But for urban we don't use it now (perhaps later with pop), so 1
    maxYield = 1.0
    demand = timeinputscalar(self.inputfolder + '/demand.tss', self.environment)

    
    # Suibility maps are calculated
    self.landUse.calculateSuitabilityMaps()

    # Allocate urban land use using demands of current time step
    self.landUse.allocate(maxYield, demand)
    self.environment = self.landUse.getEnvironment()

    # save the map of urban / non-urban
    urban = pcreq(self.environment, 1)
    self.report(urban, os.path.join(self.outputfolder,'urb'))
    
    # save the metrics
    listOfSumStats = metrics.calculateSumStats(scalar(urban), \
                                            self.sumStats, self.zones)

    j=0 # What is the aim of the j?
    for aname in self.sumStats:
        modelledmap = listOfSumStats[j]
        self.report(modelledmap, os.path.join(self.outputfolder, aname))
        j = j + 1

    for aStat in self.sumStats: # All maps should be calculated for zones
      path = generateNameT(self.outputfolder + '/' + aStat, timeStep)
      if aStat in ['np', 'pd', 'mp', 'fd']:
        # these metrics result in one value per block (here 9 blocks)
        modelledAverageArray = metrics.map2Array(path, \
                              self.inputfolder + '/sampPoint.col')
      else:
        # other metrics result in one value for the whole study area
        modelledAverageArray = metrics.map2Array(path, \
                              self.inputfolder + '/sampPointNr.col')
      # metric is saved as a list
      name1 = aStat + str(timeStep) + '.obj'
      path1 = os.path.join(self.outputfolder, name1)
      file_object1 = open(path1, 'wb')
      pickle.dump(modelledAverageArray, file_object1)
      file_object1.close()
      # the map with the metric is removed to save disk space
      os.remove(path)

    
    

############
### MAIN ###
############

nrOfTimeSteps = parameters.getNrTimesteps()
nrOfSamples = parameters.getNrSamples()
# Find the number of parameters to calibrate
nrOfParameters = len(parameters.getSuitFactorDict()[1])

# Before loop to save computation time
inputfolder = os.path.join('input_data', parameters.getCountryName())
nullMask = readmap(inputfolder + '/nullmask')

landUseList = parameters.getLandUseList()
print('Create preMCLandUse')
preMCLandUse = LandUse(landUseList, nullMask)
print('preMCLandUse created')
stations = readmap(inputfolder + '/train_stations')
preMCLandUse.determineDistanceToStations(stations)
roads = readmap(inputfolder + '/roads')
preMCLandUse.determineSpeedRoads(roads)

# Find the number of parameters to calibrate
nrOfParameters = len(parameters.getSuitFactorDict()[1])

# Before loop to save computation time
inputfolder = os.path.join('input_data', parameters.getCountryName())
nullMask = readmap(inputfolder + '/nullmask')

landUseList = parameters.getLandUseList()
preMCLandUse = LandUse(landUseList, nullMask)
stations = readmap(inputfolder + '/train_stations')
preMCLandUse.determineDistanceToStations(stations)
roads = readmap(inputfolder + '/roads')
preMCLandUse.determineSpeedRoads(roads)

# Set step size for calibration, put in parameters file?
min_p = parameters.getParametersforCalibration()[0]
max_p = parameters.getParametersforCalibration()[1]
stepsize = parameters.getParametersforCalibration()[2]

# Assure that steps in the loop have decimal place only
param_steps = np.arange(min_p, max_p + 0.1, stepsize)
for step in range(0,len(param_steps)):
    param_steps[step] = round(param_steps[step],1)

# Loop COMES HERE
sumOfParameters = 0
loopCount = 0

for p1 in param_steps:
    for p2 in param_steps:
        for p3 in param_steps:
            for p4 in param_steps:
                sumOfParameters = p1+p2+p3+p4
                if (sumOfParameters == 1):
                    loopCount = loopCount + 1
                    print('Model Run: ',loopCount,'Parameters used: ',p1,p2,p3,p4,)
                    weights = [p1,p2,p3,p4]
                    myModel = LandUseChangeModel(loopCount, weights)
                    dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
                    dynamicModel.run()

## USED TO BE THE POSTLOOP; SAVED FOR LATER USE
##print('\nrunning postmcloop...')
##print('...saving data to results folder...')
##command = "python transform_save_data.py"
##os.system(command)
##if int(self.nrSamples()) > 1:
##  print('...calculating fragstats...')		
##  command = "python plotFragstats.py"
##  os.system(command)
##  # Stochastic variables for which mean, var and percentiles are needed
##  print('...calculating statistics...')
##  names = ['urb']
##  sampleNumbers = self.sampleNumbers()
##  timeSteps = range(1, nrOfTimeSteps + 1)
##  percentiles = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
##  mcaveragevariance.mcaveragevariance(names, sampleNumbers, timeSteps)
##  names = ['ps']
##  mcpercentiles(names, percentiles, sampleNumbers, timeSteps)
##print('\n...done')
