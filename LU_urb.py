"""Land use change model of Brazil
Judith Verstegen, 2012-08-03

"""
import random
random.seed(10)
from PCRaster import *
from PCRaster.Framework import *
setrandomseed(10)
import parameters
import uncertainty
import pickle 
import covarMatrix
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
    yieldFrac -- fraction of maximum yield a cell can deliver
    forestYieldFrac -- fraction of maximum forest biomass a cell can deliver
    
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
    # This new yieldmap approach is a problem for the stochastic mode
    yieldMapName = parameters.getYieldMapName(typeNr)
    self.yieldFrac = scalar(readmap(yieldMapName))
##    self.yieldFrac = yieldFrac / 10000
    if self.typeNr == parameters.getForestNr():
      self.forest = True
    else:
      self.forest = False
    
  def setEnvironment(self, environment):
    """Update the environment (land use map)."""
    self.environment = environment

  def createInitialMask(self, globalMapNoGo, privateMapsNoGo):
    """Combine the global no-go map with areas unsuitable for this land use."""
    self.mask = globalMapNoGo
    if privateMapsNoGo is not None:
      self.mask = pcror(self.mask, privateMapsNoGo)
##        report(self.mask, 'privMask')

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
    print 'max is', float(mapmaximum(maxNr))
  
    # NEW
    # f [0,1]
    f = variableList[1]
    neighborSuitability = -1*(nrNeighborsSameLU**2) + f * 2 * maxNr *\
                               nrNeighborsSameLU
    neighborSuitability = self.normalizeMap(neighborSuitability)
##    report(neighborSuitability, 'neighborSuitability_new')
##    maxNr = ((windowLength / celllength())**2) - 1
##    report(maxNr, 'test_old')
##    neighborSuitability = nrNeighborsSameLU / maxNr
##    report(neighborSuitability, 'neighborSuitability_old')
    return neighborSuitability

  ## 2
  def getDistanceRoadSuitability(self, spreadMapRoads):
    """Return suitability map based on distance to roads."""
    variableList = self.variableDict.get(2)
    a = variableList[0]
    normalized = self.normalizeMap(spreadMapRoads)
    roadSuitability = 1 - (normalized ** a)  
##    report(roadSuitability, 'roadSuit' + str(self.typeNr))
    return roadSuitability

  ## 3
  def getDistanceWaterSuitability(self, spreadMapWater):
    """Return suitability map based on distance to water."""
    print 'check', self.variableDict
    variableList = self.variableDict.get(3)
    a = variableList[0]
    normalized = self.normalizeMap(spreadMapWater)
    waterSuitability = 1 - (normalized ** a)  
##    report(waterSuitability, 'waterSuit' + str(self.typeNr))
    return waterSuitability

  ## 4
  def getDistanceCitySuitability(self, spreadMapCities):
    """Return suitability map based on distance to large cities."""
    variableList = self.variableDict.get(4)
    a = variableList[0]
    normalized = self.normalizeMap(spreadMapCities)
    citySuitability = 1 - (normalized ** a)  
##    report(citySuitability, 'citySuit' + str(self.typeNr))
    return citySuitability

  ## 5
  def getYieldSuitability(self, yieldMap):
    """Return suitability map based on yield for crops or cattle."""
    #LOADED HERE BECAUSE OF ONEMAP
    variableList = self.variableDict.get(5)
    a = variableList[0]
    yieldSuitability = yieldMap ** a
##    yieldRelation = yexp(yieldFrac)
##    yieldSuitability = self.normalizeMap(yieldRelation)
##    report(yieldSuitability, 'yieldSuit')
    return yieldSuitability

  ## 6
  def getCurrentLandUseSuitability(self):
    """Return suitability map based on current land use type."""
    variableDict = self.variableDict.get(6) 
    current = self.nullMask
    for aKey in variableDict.keys():
      current = ifthenelse(pcreq(self.environment, aKey), \
                           variableDict.get(aKey), current)
##      print 'HERE', aKey, variableDict.get(aKey)
    currentLandUseSuitbaility = self.normalizeMap(current)
##    report(currentLandUseSuitbaility, 'autoSuit')
    return currentLandUseSuitbaility

  ## 7
  def getSlopeSuitability(self, slope):
    """Return suitability map based on slope."""
    variableList = self.variableDict.get(7)
    a = variableList[0]
    slopeSuitability = 1 - (slope ** a)
##    slopeRelation = -1 * ln(slope + 0.001)
##    slopeSuitability = self.normalizeMap(slopeRelation)
##    report(slope, 'slope')
##    report(slopeSuitability, 'slopeSuit')
    return slopeSuitability

  ## 8
  def getRandomSuitability(self):
    """Return suitability that is completely random."""
    randomSuitability = uniform(boolean(self.yieldFrac))
    return randomSuitability

  ## 9
  def getDistanceMillsSuitability(self, spreadMapMills):
    """Return suitability map based on distance to sugar cane mills."""
    variableList = self.variableDict.get(9)
    a = variableList[0]
    normalized = self.normalizeMap(spreadMapMills)
    millSuitability = 1 - (normalized ** a)  
##    report(millSuitability, 'millSuit' + str(self.typeNr))
    return millSuitability
  
  def createInitialSuitabilityMap(self, distRoads, distWater, distCities, \
                                  yieldFrac, slope, distMills):
    """Return the initial suitability map, i.e. for static factors.

    Given the maps:
    distRoads -- distances to roads
    distWater -- distances to open water
    distCities -- distances to cities
    yieldFrac -- fraction of maximum yield that can be reached in a cell
    slope -- slope (fraction, i.e. 0.12 = 12%)
    distMills -- distances to sugar cane mills
    
    Uses a lists and two dictionaries created at construction of the object:
    factors -- the names (nrs) of the suitability factors (methods) needed
    parameters -- the input parameters for those factors
    weights -- the weights that belong to those factors (how they're combined).

    """

    self.weightInitialSuitabilityMap = 0
    self.initialSuitabilityMap = spatial(scalar(0))
    i = 0
    # For every number in the suitability factor list
    # that belongs to a STATIC factor
    # the corresponding function is called providing the necessary parameters
    # and the partial suitability map is added to the total
    # taking into account its relative importance (weight)
    for aFactor in self.suitFactorList:
      if aFactor == 2:
        self.initialSuitabilityMap += self.weightList[i] * \
                                 self.getDistanceRoadSuitability(distRoads)
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor == 3:
        self.initialSuitabilityMap += self.weightList[i] * \
                                 self.getDistanceWaterSuitability(distWater)
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor == 4:
        self.initialSuitabilityMap += self.weightList[i] * \
                                 self.getDistanceCitySuitability(distCities)
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor == 5:
        self.initialSuitabilityMap += self.weightList[i] * \
                                      self.getYieldSuitability(yieldFrac)
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor == 7:
        self.initialSuitabilityMap += self.weightList[i] * \
                                 self.getSlopeSuitability(slope)
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor == 9:
        self.initialSuitabilityMap += self.weightList[i] * \
                                 self.getDistanceMillsSuitability(distMills)
        self.weightInitialSuitabilityMap += self.weightList[i]
      elif aFactor in (1, 6, 8):
        ## Dynamic factors are captured in the total suitability map
        pass
      else:
        print 'ERROR: unknown suitability factor for landuse', self.typeNr
      i += 1
    print 'weight of initial factors of', self.typeNr, \
          'is', self.weightInitialSuitabilityMap
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
      elif aFactor == 6:
        suitabilityMap += self.weightList[i] * \
                          self.getCurrentLandUseSuitability()
      elif aFactor == 8:
        suitabilityMap += self.weightList[i] * \
                          self.getRandomSuitability()
      elif aFactor in (2, 3, 4, 5, 7, 9):
        # Static factors already captured in the initial suitability map
        pass
      else:
        print 'ERROR: unknown suitability factor for landuse', self.typeNr
      i += 1
    suitabilityMap += self.weightInitialSuitabilityMap * \
                      self.initialSuitabilityMap
    self.totalSuitabilityMap = self.normalizeMap(suitabilityMap)
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
    print '\nland use type', self.typeNr
    print 'demand is:', self.demand
    if self.forest:
      print 'forest,', self.typeNr,'so remove'
      self.removeForest()
    else:
      print 'total yield is:', self.totalYield
      if self.totalYield > self.demand:
        print 'remove'
        self.remove()
      elif self.totalYield < self.demand:
        print 'add'
        self.add(immutables)
      else:
        print 'do nothing'
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
    print 'start mapMax =', float(mapMax)
    ordered = order(self.totalSuitabilityMap)
    maxIndex = int(mapmaximum(ordered))
    diff = float(self.demand - self.totalYield)
    x = int(maxIndex - diff / self.maxYield)
    xPrev = maxIndex
    i = 0
    tempEnv = self.environment
    while diff > 0 and xPrev > x:
      print 'cells to add', int(maxIndex - x)
      if x < 0:
        print 'No space left for land use', self.typeNr
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
    print 'iterations', i, 'end yield is', self.totalYield

  def remove(self):
    """Remove cells of this land use type until demand is fullfilled."""
    # Only cells already occupied by this land use can be removed
    self.totalSuitabilityMap = ifthen(self.environment == self.typeNr, \
                                      self.totalSuitabilityMap)
    ordered = order(self.totalSuitabilityMap)
    mapMin = mapminimum(self.totalSuitabilityMap)
    print 'start mapMin =', float(mapMin)
    diff = float(self.totalYield - self.demand)
    x = int(diff / (self.maxYield * 0.8))
    xPrev = 0
    i = 0
    tempEnv = self.environment
    while diff > 0 and xPrev < x and i < 100:
      print 'cells to remove', x
      # The key: cells with minimum suitability are turned into 'abandoned'
      tempEnvironment = ifthen(ordered < x, nominal(99))
      tempEnv = cover(tempEnvironment, self.environment)
      
      # Check the yield of the land use type now that less land is occupied
      self.updateYield(tempEnv)
      i += 1
      xPrev = x
      diff = float(self.totalYield - self.demand)
      if math.fmod(i, 40) == 0:
        print 'NOT getting there...'
        # Number of cells to be allocated
        x = 2 * (x + int(diff / self.maxYield))      
      else:
        # Number of cells to be allocated
        x += int(diff / self.maxYield)
    self.setEnvironment(tempEnv)
    print 'iterations', i, 'end yield is', self.totalYield
##    report(self.environment, 'newEnv' + str(self.typeNr))

  def removeForest(self):
    """Remove area of forest indicated in time series."""
    if self.demand < 0.01:
      print 'nothing to remove'
    else:
      # Only cells already occupied by this land use can be removed
      self.totalSuitabilityMap = ifthen(self.environment == self.typeNr, \
                                        self.totalSuitabilityMap)
      ordered = order(self.totalSuitabilityMap)
      mapMin = mapminimum(self.totalSuitabilityMap)
      removedBiomass = self.nullMask
      diff = 1
      tempEnv = self.environment
      print 'start mapMin =', float(mapMin)
      x = int(self.demand / self.maxYield * 0.8)
      xPrev = 0
      i = 0
      while diff > 0 and xPrev < x and i < 100:
        print 'cells to remove', x
        # The key: cells with minimum suitability are turned into 'abandoned'
        tempEnvironment = ifthen(ordered < x, nominal(98))
        tempEnv = cover(tempEnvironment, self.environment)
        removed = ifthen(tempEnvironment == 98, nominal(self.typeNr))
        # Check the yield of the land use type now that less land is occupied
        self.updateYield(removed)
        i += 1
        xPrev = x
        diff = float(self.demand - self.totalYield)
        if math.fmod(i, 40) == 0:
          print 'NOT getting there...'
          # Number of cells to be allocated
          x = 2 * (x + int(diff / self.maxYield))      
        else:
          # Number of cells to be allocated
          x += int(diff / self.maxYield)
      self.setEnvironment(tempEnv)
      print 'iterations', i, 'removed biomass is', self.totalYield

#######################################

class LandUse:
  def __init__(self, types, nullMask):
    """Construct a land use object with a nr of types and an environment."""
    self.types = types
    self.nrOfTypes = len(types)
    print '\nnr of dynamic land use types is:', self.nrOfTypes
##    self.environment = environment
    # Map with 0 in study area and No Data outside, used for cover() functions
    self.nullMask = nullMask
    self.toMeters = parameters.getConversionUnit()
    self.yearsDeforestated = nullMask
    self.forest = parameters.getForestNr()

    # maps for which no distance is needed now
    self.distWater = nullMask
    self.distRoads = nullMask

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
      
  def determineNoGoAreas(self, noGoMap, noGoLanduseList, privateNoGoSlopeDict,\
                         dem):
    """Create global no-go map, pass it to the types that add own no-go areas."""
    self.slopeMap = slope(dem)
    self.excluded = noGoMap
    privateNoGoAreas = None
    # Check the list with immutable land uses
    if noGoLanduseList is not None:
      for aNumber in noGoLanduseList:
        booleanNoGo = pcreq(self.environment, aNumber)
        self.excluded = pcror(self.excluded, booleanNoGo)
    report(scalar(self.excluded), 'excluded')
    i = 0
    for aType in self.types:
      # Get land use type specific no-go areas based on slope from dictionary
      # If not present the variable privateNoGoAreas is 'None'
      aSlope = privateNoGoSlopeDict.get(aType)
      #TURN THIS PART ON WHEN SRTM DATA PRESENT
##      if aSlope is not None:
##        privateNoGoAreas = pcrgt(self.slopeMap, aSlope)
      self.landUseTypes[i].createInitialMask(self.excluded, privateNoGoAreas)
      i += 1

  def determineDistanceToRoads(self, booleanMapRoads):
    """Create map with distance to roads, given a boolean map with roads."""
    self.distRoads = spread(booleanMapRoads, 0, 1)
    report(self.distRoads, 'distRoads.map')
    
  def determineDistanceToWater(self, booleanMapWater):
    """Create map with distance to water, given a boolean map with water."""
    self.distWater = spread(booleanMapWater, 0, 1)
    report(self.distWater, 'distWater.map')

  def determineDistanceToLargeCities(self, booleanMapCities, booleanMapRoads):
    """Create map with distance to cities, using a boolean map with cities."""
    # By using the part below one can make a map of relative time to
    # reach a city, giving roads a lower friction
    relativeSpeed = ifthenelse(booleanMapRoads, scalar(1), scalar(6))
    self.distCities = spread(booleanMapCities, 0, relativeSpeed)
    # The usual way
##    self.distCities = spread(booleanMapCities, 0, 1)
    report(self.distCities, 'distCities.map')

  def determineDistanceToRails(self, booleanMapRails):
    """Create map with distance to rails, given a boolean map with rails."""
    self.distRails = spread(booleanMapRails, 0, 1)
##    report(self.distRails, 'distRails')

  def determineDistanceToMills(self, booleanMapMills):
    """Create map with distance to rails, given a boolean map with mills."""
    self.distMills = spread(booleanMapMills, 0, 1)
    report(self.distMills, 'distMills.map')
    
  def loadDistanceMaps(self):
    """load the distance maps, when they cannot be kept in memory (fork)"""
##    print os.getcwd()
    self.distRoads = readmap('distRoads')
    self.distWater = readmap('distWater')
    self.distCities = readmap('distCities')
    self.distMills = readmap('distMills')
  
  def calculateStaticSuitabilityMaps(self, stochYieldMap):
    """Get the part of the suitability maps that remains the same."""
    for aType in self.landUseTypes:
      # Check whether the type has static suitability factors
      # Those have to be calculated only once (in initial)
      aType.createInitialSuitabilityMap(self.distRoads, self.distWater, \
                                        self.distCities, stochYieldMap, \
                                        self.slopeMap, self.distMills)

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

  def growForest(self):
    """Regrow forest at deforestated areas after 10 years."""
    # Get all cells that are abandoned in the timestep
    deforestated = pcreq(self.environment, 98)
    # Update map that counts the years a cell is deforestated
    increment = ifthen(deforestated, self.yearsDeforestated + 1)
    self.yearsDeforestated = cover(increment, self.yearsDeforestated)
    # Regrow forest after 9 years of abandonement, so it's available
    # again after 10 years
    regrown = ifthen(self.yearsDeforestated == 9, nominal(self.forest))
    reset = ifthen(regrown == nominal(self.forest), scalar(0))
    self.yearsDeforestated = cover(reset, self.yearsDeforestated)
    # Update environment
    filledEnvironment = cover(regrown, self.environment)
    self.setEnvironment(filledEnvironment)
    
  def getEnvironment(self):
    """Return the current land use map."""
    return self.environment

  def getSlopeMap(self):
    """Return the slope map."""
    return self.slopeMap
	
########################################################

class LandUseChangeModel(DynamicModel, MonteCarloModel, \
                         ParticleFilterModel):
  def __init__(self):
    DynamicModel.__init__(self)
    MonteCarloModel.__init__(self)
    ParticleFilterModel.__init__(self)
    setclone('nullMask')
##    setglobaloption('nondiagonal')

  def premcloop(self):
    # create sample points
    self.nullMask = self.readmap('nullMask')
    self.oneMask = self.readmap('oneMask')
    unique = uniqueid(self.oneMask)
    self.zones = self.readmap('zones150')
    self.samplePoints = pcreq(areamaximum(unique, self.zones) - 50, unique)
    self.noGoMap = cover(self.readmap('noGo'), boolean(self.nullMask))
    self.samplePoints = pcrand(self.samplePoints, pcrnot(self.noGoMap))
    self.samplePoints = ifthen(self.samplePoints == 1, boolean(1))
    self.samplePoints = uniqueid(self.samplePoints)
    self.report(self.samplePoints, 'sampPoint')
    self.sumStats = parameters.getSumStats()

    # attributes
    self.landuse = self.readmap('landuse2')
    self.initialSc = self.readmap('initialSc')
    roads = self.readmap('roads')
    cities = self.readmap('sp_city')
##    rails = self.readmap('railways')
    mills = readmap('mills')

    self.dem = self.readmap('dem1')
    self.yieldMap = self.readmap('rainhightechSuit1')
    
    self.roads = cover(roads, boolean(self.nullMask))
    # water map constructed from land use map
    self.water = cover(self.landuse == 5, boolean(self.nullMask))
    self.cities = cover(cities, boolean(self.nullMask))
    self.mills = cover(mills, boolean(self.nullMask))

    # List of landuse types in order of 'who gets to choose first'
    self.landUseList = parameters.getLandUseList()
    self.relatedTypeDict = parameters.getRelatedTypeDict()

    # Input values from parameters file
    # Comment out if obtained from uncertainty file
    self.suitFactorDict = parameters.getSuitFactorDict()
##    self.weightDict = Parameters.getWeightDict()
##    self.variableSuperDict = parameters.getVariableSuperDict()
    self.noGoLanduseList = parameters.getNoGoLanduseTypes()
    self.privateNoGoSlopeDict = parameters.getPrivateNoGoSlopeDict()

    # Uniform map of very small numbers, used to avoid equal suitabilities
    self.noise = uniform(1)/10000
    
    self.preMCLandUse = LandUse(self.landUseList, self.nullMask)
    self.preMCLandUse.determineDistanceToRoads(self.roads)
    self.preMCLandUse.determineDistanceToWater(self.water)
    self.preMCLandUse.determineDistanceToLargeCities(self.cities, self.roads)
    self.preMCLandUse.determineDistanceToMills(self.mills)

  def initial(self):
    # Temporary piece of code becuase of problem random seed
##    random.seed()
##    numpy.random.seed()
##    PCRaster.setrandomseed(0)
    random.seed(self.currentSampleNumber())
    np.random.seed(self.currentSampleNumber())
    setrandomseed(self.currentSampleNumber())
    
    self.uniqueNumber = self.currentSampleNumber()
    # Create the 'overall' landuse class
##    self.environment = uncertainty.getInitialLandUseMap(self.landuse)
    self.environment = cover(ifthen(self.initialSc == 1, nominal(6)),\
		      self.landuse)
    
    self.landUse = LandUse(self.landUseList, self.nullMask)
    self.landUse.setInitialEnvironment(self.environment)

    # Uncertainty that is static over time same for all lu types
    self.stochDem = uncertainty.getDem(self.dem)
    # Uncertainty that is static over time different per lu types
    self.stochYieldMap = uncertainty.getYieldMap(self.yieldMap)
    self.weightDict = uncertainty.getWeights2(self.suitFactorDict)
    self.variableSuperDict = uncertainty.getSuitabilityParameters(self.suitFactorDict)

    # Create an object for every landuse type in the list    
    self.landUse.createLandUseTypeObjects(self.relatedTypeDict, \
                                          self.suitFactorDict, \
                                          self.weightDict, \
                                          self.variableSuperDict, \
                                          self.noise)

    # Static suitability factors
    self.landUse.determineNoGoAreas(self.noGoMap, self.noGoLanduseList, \
                                    self.privateNoGoSlopeDict, self.stochDem)

    # population, cattle etc taken away
##    self.landUse.determineDistanceToRoads(self.roads)
##    self.landUse.determineDistanceToWater(self.water)
##    self.landUse.determineDistanceToLargeCities(self.cities, self.roads)
##    self.landUse.determineDistanceToMills(self.mills)
    self.landUse.loadDistanceMaps()
    self.landUse.calculateStaticSuitabilityMaps(self.stochYieldMap)

          
    # Draw random numbers between zero and one
    # To determine yield and demand
    self.demandStoch = round(float(mapnormal()),2)
    print 'FRACTION DEMAND IS',self.demandStoch,'\n'
    self.maxYieldStoch = mapuniform()
    self.bioMaxYieldStoch = mapuniform()

  def dynamic(self):
    timeStep = self.currentTimeStep()
    print '\ntime step', timeStep
    
    # NEW: report the state of the previous time step
    sugarCane = pcreq(self.environment, 6)
#    self.report(sugarCane, 's_prev')
    
    # Get max yield and demand per land use type
    # But for Brazil we don't use it, so 1
##    maxYieldUp = timeinputscalar('maxYieldUp.tss', self.environment)
##    maxYieldLow = timeinputscalar('maxYieldLow.tss', self.environment)
##    maxYieldDiff = (maxYieldUp - maxYieldLow)
##    maxYield = maxYieldDiff * self.maxYieldStoch + maxYieldLow
    maxYield = 1.0
    
##    demandUp = timeinputscalar('demandUp.tss', self.environment)
##    demandLow = timeinputscalar('demandLow.tss', self.environment)
    demandAv = timeinputscalar('demand_av.tss', self.environment)
    demandSd = spatial(scalar(0))#timeinputscalar('demand_sd.tss', self.environment)
    demand = demandAv + self.demandStoch * demandSd
    
    # Suibility maps are calculated
    self.landUse.calculateSuitabilityMaps()

    # Allocate new land use using demands of current time step
    self.landUse.allocate(maxYield, demand)
    self.landUse.growForest()
    self.environment = self.landUse.getEnvironment()

    # reporitings
##    self.report(self.environment, 'landUse')
##    os.system('legend --clone landuse.map -f \"legendLU.txt\" %s ' \
##              %generateNameST('landUse', self.currentSampleNumber(),timeStep))
    sugarCane = pcreq(self.environment, 6)
##    if not timeStep in [9,11,13,15,17,19]:
    self.report(sugarCane, 'sSc')
##      self.report(scalar(sugarCane), 'sSc')

    # Objects with pickle or marshal
    name1 = 'weights' + str(timeStep) + '.obj'
    path1 = os.path.join(str(self.currentSampleNumber()), name1)
##    path = str(self.currentSampleNumber()) + '\stateVar' + '\\weights.obj'
    file_object1 = open(path1, 'w')
    pickle.dump(self.weightDict, file_object1)
    file_object1.close()

    name2 = 'superDict' + str(timeStep) + '.obj'
    path2 = os.path.join(str(self.currentSampleNumber()), name2)
    file_object2 = open(path2, 'w')
    pickle.dump(self.variableSuperDict, file_object2)
    file_object2.close()

    name3 = 'demandStoch' + str(timeStep) + '.obj'
    path3 = os.path.join(str(self.currentSampleNumber()), name3)
    file_object3 = open(path3, 'w')
    pickle.dump(self.demandStoch, file_object3)
    file_object3.close()

    name4 = 'number' + str(timeStep) + '.obj'
    path4 = os.path.join(str(self.currentSampleNumber()), name4)
    file_object4 = open(path4, 'w')
    pickle.dump(self.uniqueNumber, file_object4)
    file_object4.close()
    
    # save the sum stats of the calibration blocks
    listOfSumStats = covarMatrix.calculateSumStats(scalar(sugarCane), \
                                            self.sumStats, self.zones)
    modelledAverageMap = listOfSumStats[0]
    modelledPatchNumber = listOfSumStats[1]
    modelledLS = listOfSumStats[2]
    self.report(modelledAverageMap, 'av')
    self.report(modelledPatchNumber, 'nr')
    self.report(modelledLS, 'ls')

    for aStat in self.sumStats:
      path = generateNameST(aStat, self.currentSampleNumber(),timeStep)
      if aStat == 'av':
        modelledAverageArray = covarMatrix.map2Array(path, 'sampPointAvSelection.col')
      else:
        modelledAverageArray = covarMatrix.map2Array(path, 'sampPointNrSelection.col')
      name1 = aStat + str(timeStep) + '.obj'
      path1 = os.path.join(str(self.currentSampleNumber()), name1)
      file_object1 = open(path1, 'w')
      pickle.dump(modelledAverageArray, file_object1)
      file_object1.close()
      os.remove(generateNameST(aStat,self.currentSampleNumber(), timeStep))

    # save the sumstats of the validation blocks
    listOfSumStats = covarMatrix.calculateSumStats(scalar(sugarCane), \
                                          self.sumStats, self.zones, True)
    modelledAverageMap = listOfSumStats[0]
    modelledPatchNumber = listOfSumStats[1]
    modelledLS = listOfSumStats[2]
    self.report(modelledAverageMap, 'av')
    self.report(modelledPatchNumber, 'nr')
    self.report(modelledLS, 'ls')

    for aStat in self.sumStats:
      path = generateNameST(aStat, self.currentSampleNumber(),timeStep)
      if aStat == 'av':
        modelledAverageArray = covarMatrix.map2Array(path, 'sampPointAvValidation.col')
      else:
        modelledAverageArray = covarMatrix.map2Array(path, 'sampPointNrValidation.col')
      name1 = aStat + '_val' + str(timeStep) + '.obj'
      path1 = os.path.join(str(self.currentSampleNumber()), name1)
      file_object1 = open(path1, 'w')
      pickle.dump(modelledAverageArray, file_object1)
      file_object1.close()
      os.remove(generateNameST(aStat,self.currentSampleNumber(), timeStep))
      
  def postmcloop(self):
    print '\nrunning postmcloop...'
    if int(self.nrSamples()) > 1:
      print '...calculating weights...'
      command = "python get2_extra.py"
      os.system(command)
      print '...calculating fragstats...'			
      command = "python postloop_frst.py"
      os.system(command)
      command = "python postloop_frst_val.py"
      os.system(command)
      # Stochastic variables for which mean, var and percentiles are needed
      print '...calculating statistics...'
##      names = ['sSc', 'nr', 'av']
      names = ['sSc']
      sampleNumbers = self.sampleNumbers()
      timeSteps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] #self.timeSteps()
      percentiles = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
      mcaveragevariance.mcaveragevariance(names, sampleNumbers, timeSteps)
##      names = ['ps']
##      mcpercentiles(names, percentiles, sampleNumbers, timeSteps)
    print '\n...done'

    
  def updateWeight(self):
    modelledData = self.readmap('sSc')
    listOfSumStats = covarMatrix.calculateSumStats(modelledData, \
                                                    self.sumStats, self.zones)
    modelledAverageMap = listOfSumStats[0]
##    report(modelledAverageMap, 'test1')
    modelledPatchNumber = listOfSumStats[1]
    modelledPatchMap = listOfSumStats[2]

    # Be aware, without covar matrix has not been tested anymore since long!
    if parameters.getCovarOn() == 0:
      # Read sum stats from file
      observedAverageMap = self.readDeterministic('av_ave')
      observedNumberMap = self.readDeterministic('nr_ave')
      observedPatchMap = self.readDeterministic('ls_ave')
##      observedAverageMap = self.readDeterministic('obs')  
      #sc = scalar(self.readDeterministic('sc'))
      #listOfSumStats = covarMatrix.calculateSumStats(sc, self.sumStats,\
                                                      #self.zones)
      #observedAverageMap = listOfSumStats[0]
      #observedPatchNumber = listOfSumStats[1]
      #observedPatchMap = listOfSumStats[2]
      
      observedAveragePoints = ifthen(self.samplePoints, observedAverageMap)

      observedStdDevPoints1 = ifthenelse(observedAveragePoints > 0, \
                                      observedAveragePoints * scalar(0.6), 0.1)
#     self.report(observedStdDevPoints, 'sd')
      total1 = maptotal(((observedAveragePoints - modelledAveragePoints)\
                      ** 2) / (2.0 * (observedStdDevPoints1 ** 2)))
        
      # How to combine the two measures here?
      # Can they cancel each other out now (should not)
      weight = exp(0.0 - (total1))
      weightFloatingPoint, valid = cellvalue(weight, 1, 1)
      print 'TOTAL is', float(total1), float(total2), \
          'WEIGHT is', weightFloatingPoint
    
    else:
      # Read sum stats from file
      observedAverageMap = self.readDeterministic('av_ave_c')
      observedNumberMap = self.readDeterministic('nr_ave_c')
      observedPatchMap = self.readDeterministic('ls_ave_c')      
      
      # or calculate sum stats from original (system state) maps
      #sc = scalar(self.readDeterministic('sc'))
      #listOfSumStats = covarMatrix.calculateSumStats(sc, self.sumStats,\
                                                      #self.zones)
      #observedAverageMap = listOfSumStats[0]
      #observedNumberMap = listOfSumStats[1] 
      #observedPatchMap = listOfSumStats[2]

      # Here selection
      path = 'covar000.00' + str(self.currentTimeStep())
      covarObsErr = numpy.loadtxt(path)
##      print covarObsErr * 100
##      print covarObsErr.shape[0]
##      print covarObsErr.shape[1]
      b = np.matrix(covarObsErr*1).I
##      b = np.invert(np.matrix(covarObsErr))
      inverseCovar = np.array(b)
      
      # Difference observations and model output
      obsMinusModel1 = observedAverageMap - modelledAverageMap
      report(obsMinusModel1, 'test')
      matrix1 = covarMatrix.map2Array('test', 'sampPointAvSelection.col')
##      print matrix1.shape[1]

      obsMinusModel2 = observedNumberMap - modelledPatchNumber
      report(obsMinusModel2, 'test')
      matrix2 = covarMatrix.map2Array('test', 'sampPointNrSelection.col')
      
      obsMinusModel3 = observedPatchMap - modelledPatchMap
      report(obsMinusModel3, 'test')
      matrix3 = covarMatrix.map2Array('test', 'sampPointNrSelection.col')

##      obsMinusModel = numpy.concatenate((matrix1, matrix2))
      obsMinusModel = numpy.append(matrix1, matrix2)
      obsMinusModel = numpy.append(obsMinusModel, matrix3)
##      obsMinusModel = matrix1
##      print obsMinusModel
      firstTerm = numpy.dot(obsMinusModel.T, inverseCovar)
      total = 0.0 - (numpy.dot(firstTerm, obsMinusModel) / 2.0)
      

      weight = exp(total)
      weightFloatingPoint, valid = cellvalue(weight, 1, 1)
      # dedented out because of python version
      # I think it was for weight 0 all
##      if math.isinf(weightFloatingPoint):
##        print 'inf'
##        weightFloatingPoint = 0.0

##      weightFloatingPoint = float(weight)
      print 'TOTAL is', float(total), \
           'WEIGHT is', weightFloatingPoint
      
    
    return weightFloatingPoint

  def suspend(self):
    print 'SUSPEND', str(self.currentSampleNumber()), '\n'
    # Maps
    self.reportState(self.stochYieldMap, 'yield')
    self.reportState(self.environment, 'env')
    self.reportState(self.stochDem, 'dem')
    
    # Objects with pickle or marshal
    path1 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'weights.obj')
    file_object1 = open(path1, 'w')
    pickle.dump(self.weightDict, file_object1)
    file_object1.close()

    path2 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'superDict.obj')
    file_object2 = open(path2, 'w')
    pickle.dump(self.variableSuperDict, file_object2)
    file_object2.close()

    path3 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'demandStoch.obj')
    file_object3 = open(path3, 'w')
    pickle.dump(self.demandStoch, file_object3)
    file_object3.close()

    path4 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'number.obj')
    file_object4 = open(path4, 'w')
    pickle.dump(self.uniqueNumber, file_object4)
    file_object4.close()

  def resume(self):
    print 'RESUME', str(self.currentSampleNumber())
    # max yield stoch does not matter now!
    self.maxYieldStoch = 1
    
    # Maps
    self.stochYieldMap = self.readState('yield')
    self.environment = self.readState('env')
    self.stochDem = self.readState('dem')
    
    print '\n...', float(maptotal(self.stochYieldMap))
    
    # Objects with pickle or marshal
    path1 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'weights.obj')
    filehandler1 = open(path1, 'r') 
    self.weightDict = pickle.load(filehandler1) 
    print self.weightDict
    filehandler1.close()

    path2 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'superDict.obj')
##    path2 = str(self.currentSampleNumber()) + '\stateVar' + '\\superDict.obj'
    filehandler2 = open(path2, 'r') 
    self.variableSuperDict = pickle.load(filehandler2) 
    print self.variableSuperDict
    filehandler2.close()

    path3 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'demandStoch.obj')
    filehandler3 = open(path3, 'r') 
    self.demandStoch = pickle.load(filehandler3) 
    print 'stoch demand is', self.demandStoch
    filehandler3.close()

    path4 = os.path.join(str(self.currentSampleNumber()), 'stateVar', \
                        'number.obj')
    filehandler4 = open(path4, 'r') 
    self.uniqueNumber = pickle.load(filehandler4) 
    filehandler4.close()
   
    self.landUse = LandUse(self.landUseList, self.nullMask)
    self.landUse.setInitialEnvironment(self.environment)
    self.landUse.loadDistanceMaps()
    self.landUse.createLandUseTypeObjects(self.relatedTypeDict, \
                                      self.suitFactorDict, \
                                      self.weightDict, \
                                      self.variableSuperDict, \
                                      self.noise)
    self.landUse.setEnvironment(self.environment)
    self.landUse.determineNoGoAreas(self.noGoMap, self.noGoLanduseList, \
                                    self.privateNoGoSlopeDict, self.stochDem)
    self.landUse.calculateStaticSuitabilityMaps(self.stochYieldMap)
    

nrOfTimeSteps = parameters.getNrTimesteps()
nrOfSamples = parameters.getNrSamples()
myModel = LandUseChangeModel()
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
mcModel = MonteCarloFramework(dynamicModel, nrOfSamples)
mcModel.setForkSamples(True,16)
##mcModel.run()
pfModel = SequentialImportanceResamplingFramework(mcModel)
##pfModel = ResidualResamplingFramework(mcModel)
pfModel.setFilterTimesteps([3,4])
##pfModel.setFilterTimesteps([3,4,5])
pfModel.run()
