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
import time

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
    self.noise = noise
    self.nullMask = nullMask

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
    """Return suitability map based on nr of neighbors with a related type."""
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
    # Test of reversing the sign of the square function coefficient
    neighborSuitability = 1*(nrNeighborsSameLU**2) + f * 2 * maxNr *\
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
    """Return suitability map based on distance to train stations."""
    variableList = self.variableDict.get(2)
    a = variableList[0]
    normalized = self.normalizeMap(spreadMap)
    roadSuitability = 1 - (normalized ** a)  
    report(roadSuitability, 'suit_station' + str(self.typeNr))
    return roadSuitability

  ## 3
  def getTravelTimeCityBorder(self):
    """Return suitability map based on distance to the largest city."""
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

  def getRandomClumps(self):
    """Return random map."""
    # Create map of random clumps
    randmap = windowaverage(uniform(boolean(self.yieldFrac)), \
                            10 * celllength())
    # Add randomness parameter from model by Garcia. Alfa (0,1) reflects the randomness level.
    # In the noise (uniform) map change the zero values to very small values, to allow logarythmic function.
    # Total suitability is multiplied by the randomness value v.
    alpha = parameters.getAlphaValue()
    # Replace the value of zero to a very small value
    v = ifthenelse(self.noise==0,1E-300, self.noise)
    # Apply function by White
    v = 1 + ((-ln(v))**alpha)
    '''# Apply function by Garcia
    v = exp(-alpha*(1-v))
    # Apply randomness factor to the map with random clumps to make them less dense
    randomClumps = randmap * v
    randomClumps = self.normalizeMap(randomClumps)
    report(randomClumps, 'rand_clumps.map')
    
    return randomClumps'''
    return v
 
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
    #self.initialSuitabilityMap += self.noise <==== instead trying to add noise in total suitability
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
    suitabilityMap += self.weightInitialSuitabilityMap * \ # WHY MULTIPLY AGAIN BY WEIGHT
                      self.initialSuitabilityMap
    # Add randomness
    suitabilityMap = self.getRandomClumps() * suitabilityMap
    # Normalize the total suitability map and report
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
    #print('\nnr of dynamic land use types is:', self.nrOfTypes)
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
    pName = 'parameters_iteration_' + str(nr) + '.obj'
    pPath = os.path.join(self.outputfolder, pName)
    parametersFile = open(pPath, 'wb')
    pickle.dump(weights, parametersFile)
    parametersFile.close()

  def initial(self):
    # create sample points
    self.nullMask = self.readmap(self.inputfolder + '/nullmask')
    self.oneMask = self.readmap(self.inputfolder + '/onemask')
    # load a map with random uniform values
    self.uniformMap = self.readmap(self.inputfolder + '/uniform')
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
    self.calibrationMask = self.readmap(self.inputfolder + '/zones_calibration')
    self.validationMask = self.readmap(self.inputfolder + '/zones_validation')

    # List of landuse types in order of 'who gets to choose first'
    self.landUseList = parameters.getLandUseList()
    self.relatedTypeDict = parameters.getRelatedTypeDict()

    # Input values from parameters file
    self.suitFactorDict = parameters.getSuitFactorDict()
    self.variableSuperDict = parameters.getVariableSuperDict()
    self.noGoLanduseList = parameters.getNoGoLanduseTypes() 

    # Uniform map of small numbers, used to avoid equal suitabilities.
    # The same uniform map is applied in each iteration.
    self.noise = self.uniformMap # noise added as a uniform map created in the create_initial_maps.py
    
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

    # Select the urban areas only for the calibration and validation area
    urban_cal = ifthen(self.calibrationMask, urban)
    urban_val = ifthen(self.validationMask, urban)
    
    # save the metrics
    listOfSumStats = metrics.calculateSumStats(scalar(urban), \
                                            self.sumStats, self.zones)               

    # Save the metrics for calibration and validation based on the area
    # For each meric a value is saved for the preselected cell(s).
    # The cell coordinates are created in create_initial_maps.py and saved in .col files:
    # to be checked: each landscape metrics can be saved in sampPoint.col, probably also the _cal and _val.
    # RMSE get the mean value for all zones. RMSE for 16 same zones == RMSE for 8 same zones
    col_files = {
      'fdi': 'sampPoint.col',         # each zone
      'wfdi': 'sampPoint.col',        # each zone
      'cilp': 'sampSinglePoint.col',  # single point in the middle of the study area
      'pd': 'sampSinglePoint.col',    # single point in the middle of the study area
      'urb': 'sampPointNr.col',       # point for each cell
      'cilp_cal': 'sampPoint_cal.col',# point for each calibration zones
      'cilp_val': 'sampPoint_val.col',# point for each validation zones
      'pd_cal': 'sampPoint_cal.col',  # point for each calibration zones
      'pd_val': 'sampPoint_val.col'   # point for each validation zones
      }
    
    j=0
    part_metrics = []

    # First, calculate the statistics and create the maps:
    for aname in self.sumStats:
      modelledmap = listOfSumStats[j]
      self.report(modelledmap, os.path.join(self.outputfolder, aname))
      j = j + 1

      # Include selected zones for metrics that are calculated for the whole map (pd)
      # or for the biggest patch only (cilp)
      if aname in ['cilp','pd']:
        stat_cal = metrics.calculateSumStats(scalar(urban_cal), \
                                            [aname], self.zones)
        stat_val = metrics.calculateSumStats(scalar(urban_val), \
                                            [aname], self.zones)
        self.report(stat_cal[0], os.path.join(self.outputfolder, aname+'_cal'))
        self.report(stat_val[0], os.path.join(self.outputfolder, aname+'_val'))
        part_metrics.append(aname+'_cal')
        part_metrics.append(aname+'_val')
      
    # Then save the metrics and urban areas as pickle objects
    for aStat in self.sumStats + ['urb'] + part_metrics:
      path = generateNameT(self.outputfolder + '/' + aStat, timeStep)
      modelledAverageArray = metrics.map2Array(path, self.inputfolder + '/' + col_files[aStat])    
      # metric and urban areas are saved as a list
      name1 = aStat + str(timeStep) + '.obj'
      path1 = os.path.join(self.outputfolder, name1)
      file_object1 = open(path1, 'wb')
      pickle.dump(modelledAverageArray, file_object1)
      file_object1.close()
      # the map with the metric or urban area is removed to save disk space
      os.remove(path)
    

############
### MAIN ###
############

start_time = time.time()
nrOfTimeSteps = parameters.getNrTimesteps()
#nrOfSamples = parameters.getNrSamples() # This variable is not being used as MC was eliminated from the model
# Find the number of parameters to calibrate
nrOfParameters = len(parameters.getSuitFactorDict()[1])
nrOfIterations = parameters.getNumberofIterations()

# Before loop to save computation time
inputfolder = os.path.join('input_data', parameters.getCountryName())
nullMask = readmap(inputfolder + '/nullmask')

landUseList = parameters.getLandUseList()
preMCLandUse = LandUse(landUseList, nullMask)
stations = readmap(inputfolder + '/train_stations')
preMCLandUse.determineDistanceToStations(stations)
roads = readmap(inputfolder + '/roads')
preMCLandUse.determineSpeedRoads(roads)


#######################
### Loop COMES HERE ###
#######################

loopCount = 0

# Set step size for calibration
min_p = parameters.getParametersforCalibration()[0]
max_p = parameters.getParametersforCalibration()[1]
stepsize = parameters.getParametersforCalibration()[2]

# Assure that steps in the loop have 3 decimal place only
param_steps = np.around(np.arange(min_p, max_p + 0.100, stepsize),decimals=3)

# Run the model
print('\n################################################')
print('Run LU_urb model')
print('Number of iterations: ', nrOfIterations)
print('Number of time steps: ', nrOfTimeSteps)
print('Min parameter value: ', min_p, '. Max parameter value: ', max_p,'. Parameter steps: ', param_steps)


for p1,p2,p3,p4 in ((a,b,c,d) for a in param_steps for b in param_steps for c in param_steps for d in param_steps):
  sumOfParameters = p1+p2+p3+p4
  if (sumOfParameters > 0.9999 and sumOfParameters < 1.0001):
      loopCount = loopCount + 1
      print('\n################################################')
      print('Model Run: ',loopCount,'. Parameters used: ',p1,p2,p3,p4)
      myModel = LandUseChangeModel(loopCount, [p1,p2,p3,p4])
      dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
      dynamicModel.run()                  

print('\n################################################')
print("--- Number of iterations of a loop: %s ---" % (nrOfIterations))
print("--- Number of timesteps: %s ---" % (nrOfTimeSteps))
print("--- Program execution: ", str(int((time.time() - start_time))/60), " minutes ---")


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
