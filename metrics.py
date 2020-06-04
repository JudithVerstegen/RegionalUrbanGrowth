import string
import os, shutil, numpy
#import generalfunctions
import matplotlib.pyplot as plt
from pcraster import *
from pcraster.framework import *
import parameters

inputfolder = os.path.join(os.getcwd(), 'input_data', parameters.getCountryName())

def map2Array(filename, rowColFile):
  """Selects values at row, col from raster name in Monte Carlo samples.

  filename -- Name of raster.
  rowColFile -- File with row and col index of cell to read.
  The returned array does not contain missing values so the size is minimal
  sampleNumbers but possibly smaller.

  Returned array has elements of type numpy.float32"""
  sampleFile = open(rowColFile, 'r')
  samplePoints = sampleFile.readlines()
  sampleFile.close()
  amap = readmap(filename)
##  mask = numpy.zeros((1, len(samplePoints))).astype(numpy.bool_)
##  array = numpy.zeros((1, len(samplePoints))).astype(numpy.float32)
  mask = numpy.zeros(len(samplePoints)).astype(numpy.bool_)
  array = numpy.zeros(len(samplePoints)).astype(numpy.float32)
  j = 0
  for point in samplePoints:
    attributes = point.split()
##    print(attributes)
    row = math.ceil(float(attributes[1]))
    col = math.ceil(float(attributes[0]))
##    print(row, col)
    array[j], mask[j] = cellvalue(amap, row, col)
    if mask[j] == False:
      array[j] = numpy.nan
    j += 1

  #array = numpy.compress(mask, array)
  #array = numpy.ma.array(array, mask = mask) <-- didn't work
  #print(numpy.unique(array))
  return array

def mySelectSArray(name, sampleNumbers, rowColFile, base=None):
  """Selects values at row, col from raster name in Monte Carlo samples.

  name -- Name of raster.
  sampleNumber -- Numbers of MC samples to use.
  rowColFile -- File with row and col index of cell to read.
  The returned array does not contain missing values so the size is maximimal
  sampleNumbers but possibly smaller.

  Returned array has elements of type numpy.float32"""

  sampleFile = open(rowColFile, 'r')
  samplePoints = sampleFile.readlines()
  sampleFile.close()
  mask = numpy.zeros((len(samplePoints), len(sampleNumbers))).astype(numpy.bool_)
  array = numpy.zeros((len(samplePoints), len(sampleNumbers))).astype(numpy.float32)
  i = 0
  while i < len(sampleNumbers):
    filename = generateNameS(name, sampleNumbers[i])
    if base is not None:
      filename = os.path.join(base,filename)
      #if i == 0: aguila(filename)
    amap = readmap(filename)
    j = 0
    for point in samplePoints:
      attributes = string.split(point)
      row = math.ceil(float(attributes[1]))
      col = math.ceil(float(attributes[0]))
      array[[j], [i]], mask[[j], [i]] = cellvalue(amap, row, col)
      j += 1
    i += 1
#  array = numpy.compress(mask, array)
  print('array', array)
  return array

def selectSArrayMultipleRasters(names,sampleNumbers,rowColFiles, base=None):
  """Selects at row, col from each raster name
  Returned array is 'nested', i.e. each element contains
  an array with the values of a raster name"""
  a = []
  i = 0
  for name in names:
    arrayOfRaster = mySelectSArray(name,sampleNumbers,rowColFiles[i], base)
    a.append(arrayOfRaster)
    i += 1
  c = numpy.vstack(a)
  return c

def calculateSumStats(systemState, listOfSumStats, zones, validation=False):
  """Return a list of sum stat maps for the sum stat"""
  listOfMaps = []
  ##  mask = readmap(inputfolder + '/zones_selection')
  ##  if validation == True: mask = readmap(inputfolder + '/zones_validation')
  ##  systemState = ifthen(mask, systemState)

  # Get values common for more than one metric
  unique = uniqueid(boolean(spatial(scalar(1))))
  clumps = ifthen(boolean(systemState) == 1, clump(boolean(systemState)))
  zero_mask = ifthen(defined(systemState),spatial(nominal(0)))
  #numberMap = areadiversity(clumps, spatial(nominal(1))) # doesnt work for test map
  oneCellPerPatch = pcreq(areamaximum(unique, clumps), unique) # gets the cell in the right bottom corner of a patch
  scNegative = ifthenelse(boolean(systemState) == 1, boolean(0), boolean(1))
  borders = ifthen(boolean(systemState) == 1, window4total(scalar(scNegative)))
  perimeter = areatotal(borders, nominal(clumps))# in cell units (100m)
  patchSizes = areaarea(clumps)/cellarea() # size of the clumps in cell units (10 000m2)
  zone_area = areaarea(zones)/cellarea() # size of the zones in cell units (10 000m2)
  map_area = areaarea(zero_mask)/cellarea() # size of the whole study area in cell units (10 000m2)

  for aStat in listOfSumStats:
    if aStat == 'np': # Number of patches in one zone
      average_nr = cover(areadiversity(clumps, zones), spatial(scalar(0)))  
      listOfMaps.append(average_nr)
    elif aStat == 'pd': # Patch density in a whole study area
      patches_nr = areadiversity(clumps, zero_mask)
      patch_density = patches_nr/map_area
      listOfMaps.append(patch_density)
    elif aStat == 'mp': # Mean patch size in a zone.
      # If patch is in more than one zone it is assigned to one zone only...
      patchSizeOneCell = ifthen(oneCellPerPatch, patchSizes)
      averagePatchSize = areaaverage(patchSizeOneCell, zones)
      averagePatchSizeScalar = cover(averagePatchSize, spatial(scalar(0)))
      listOfMaps.append(averagePatchSizeScalar)
    elif aStat == 'fdi': # Mean fractal dimension index of patches in one zone.
      ### Value of the metric is dependent on the unit used
      ### 'perimeter' and 'patchSizes' need to be higher than e = ~2.71
      fractalDimension = 2*ln(0.25 * perimeter)/ln(patchSizes)
      patchFractalDimensionOneCell = ifthen(oneCellPerPatch, fractalDimension)
      averageFractalDimension = areaaverage(patchFractalDimensionOneCell, zones)
      listOfMaps.append(averageFractalDimension)
    elif aStat == 'cilp': # Compactness index of the largest patch (CILP)
      # This metric is calculated for one patch only.
      # Saved as a one value for the whole map
      biggestPatchSize = mapmaximum(patchSizes)/cellarea() # largest patch area in the cell unit
      biggestPatchSize = ifthen(defined(systemState), biggestPatchSize)
      biggestPatchPerimeter = mapmaximum(
        ifthen(patchSizes == biggestPatchSize, perimeter)) # perimeter of the largest patch
      biggestPatchPerimeter = ifthen(defined(systemState),biggestPatchPerimeter)
      CILP = (2 * numpy.pi * sqrt(biggestPatchSize / numpy.pi)) / biggestPatchPerimeter
      listOfMaps.append(CILP)
    elif aStat == 'wfdi': # Area weighted mean patch fractal dimension index in one zone
      wFractalDimensionIndex = (2*ln(0.25 * perimeter)/ln(patchSizes))*(patchSizes/zone_area)
      wFractalDimensionIndexOneCell = ifthen(oneCellPerPatch, wFractalDimensionIndex)
      WFDI = areaaverage(wFractalDimensionIndexOneCell, zones)
      listOfMaps.append(WFDI)
    elif aStat == 'cohesion': # Patch Cohesion Index in a zone
      ### measures the physical connectedness of the corresponding patch type
      summedPerimeter = areatotal(perimeter, zones) # in cell units
      summedPerimeterArea = areatotal(perimeter * sqrt(patchSizes),zones)
      cohesion = (1 - (summedPerimeter/summedPerimeterArea))*(1-1/sqrt(zone_area))
      listOfMaps.append(cohesion)
    elif aStat == 'ed': # Edge Density in a zone
      clumps_edges = perimeter/patchSizes
      ed = maptotal(clumps_edges)/map_area
      listOfMaps.append(ed)
    elif aStat == 'lpi': # Largest Patch Index in a whole study area
      # This metric is calculated for one patch only.
      # Saved as a one value for the whole map
      biggestPatchSize = mapmaximum(patchSizes) # largest patch area in the cell unit
      LPI = biggestPatchSize/map_area # largest patch area in the cell unit
      listOfMaps.append(LPI)
    elif aStat == 'contagion': # Contagion Index in a zone
      # ratio of the the observed contagion to the maximum possible contagion for the given number of LU types
      P_urb = areatotal(ifthen(boolean(clumps),scalar(1))/zone_area,zones)# proportion of the selected land use type in a zone
      P_nonurb = 1-P_urb
      urb_map = ifthen(boolean(systemState) == 1, scalar(1))
      nonurb_map = ifthen(boolean(systemState) == 0, scalar(1))
      # Calculate number of joints between cells for a zone, depending on the land use types
      g_urb_urb = areatotal(ifthen(boolean(urb_map),window4total(urb_map)),zones)
      g_urb_nonurb = areatotal(ifthen(boolean(urb_map),window4total(nonurb_map)),zones)
      g_nonurb_urb = areatotal(ifthen(boolean(nonurb_map),window4total(urb_map)),zones)
      g_nonurb_nonurb = areatotal(ifthen(boolean(nonurb_map),window4total(nonurb_map)),zones)
      # Calculate components of CONTAG metric, depending on the land use types
      c_urb_urb = P_urb * g_urb_urb / (g_urb_urb + g_urb_nonurb)
      c_urb_nonurb = P_urb * g_urb_nonurb / (g_urb_urb + g_urb_nonurb)
      c_nonurb_urb = P_nonurb * g_nonurb_urb / (g_nonurb_urb + g_nonurb_nonurb)
      c_nonurb_nonurb = P_nonurb * g_nonurb_nonurb / (g_nonurb_urb + g_nonurb_nonurb)
      CONTAG = 1 + (c_urb_urb*ln(c_urb_urb) + c_urb_nonurb*ln(c_urb_nonurb) + c_nonurb_urb*ln(c_nonurb_urb)\
                    +c_nonurb_nonurb*ln(c_nonurb_nonurb)) / (2 * ln(2))
      listOfMaps.append(CONTAG)
    else:
      print('ERRRRRRRRRRRROR, unknown sum stat')
  return listOfMaps

def makeCalibrationMask(rowColFile, zoneMap):
  """Return mask that excludes non-calibration areas + update covar matrix"""
  # The list of blocks that should be included in the draw
  # Now the 10 that were selected before (little no go and MV)
  # RowColFile should include x y and block number (in zones.map)
  textfile = open(rowColFile, 'r')
  listOfBlocks = []
  for aLine in textfile:
    columns = aLine.split()
##    print columns
    listOfBlocks.append(columns)
  textfile.close()
##  print listOfBlocks

  # half of them is for calibration and half for validation
  length = len(listOfBlocks)
  # Draw using indici, because indici are required to adapt covar matrix
  # First we used half of the blocks, len(listOfBlocks)/2, now 3
  listOfIndici = range(0, length)
  selection = random.sample(listOfIndici, length/2)
  selection.sort()
  print(selection)
  # make a new row col file for the blocks
  newTextfile1 = open(inputfolder + '/sampPointAvSelection.col', 'w')
  newTextfile2 = open(inputfolder + '/sampPointNrSelection.col', 'w')
  lookuptable1 = open(inputfolder + '/lookupTable_cal.tbl', 'w')
  
  newTextfile3 = open(inputfolder + '/sampPointAvValidation.col', 'w')
  newTextfile4 = open(inputfolder + '/sampPointNrValidation.col', 'w')
  lookuptable2 = open(inputfolder + '/lookupTable_val.tbl', 'w')
  i = 0
  j = 0
  for aNumber in listOfIndici:
    aBlock = listOfBlocks[aNumber]
    if aNumber in selection:
      lookuptable1.write(aBlock[-1] + ' ' + str(1) + '\n')
      for anItem in aBlock:
        newTextfile1.write(anItem + ' ')
        if i == 0:
          newTextfile2.write(anItem + ' ')
      newTextfile1.write('\n')
      i += 1
    else:
      lookuptable2.write(aBlock[-1] + ' ' + str(1) + '\n')
      for anItem in aBlock:
        newTextfile3.write(anItem + ' ')
        if i == 0:
          newTextfile4.write(anItem + ' ')
      newTextfile3.write('\n')
      i += 1
  newTextfile1.close()
  newTextfile2.close()
  lookuptable1.close()
  lookuptable2.close()

  # Make the mask, i.e. blocks that are SELECTED
  blocksTrue = lookupboolean(inputfolder + '/lookupTable_cal.tbl', zoneMap)
  report(blocksTrue, inputfolder + '/zones_selection.map')
  blocksTrue = lookupboolean(inputfolder + '/lookupTable_val.tbl', zoneMap)
  report(blocksTrue, inputfolder + '/zones_validation.map')


# TEST
""" Testing on the map with one zone: size 30 km x 30 km, with three patches: 700 km2, 200 km2, 100 km2 """
test_map = os.path.join(os.getcwd(), 'data', 'test_data', 'metric_test_3patches_IE.map')
systemState = readmap(test_map) == 1 # select urban or predefined pattern
zones_map = os.path.join(inputfolder, 'zones.map')
#mask = os.path.join(inputfolder, 'zones_calibration.map')
zones = readmap(zones_map)
# use a mask
#systemState = ifthen(mask, systemState)

# put HERE the name(s) of the metric(s) you want to test
# ['np', 'pd', 'mp', 'fdi', 'wfdi', 'cilp','cohesion','ed']
metrics = ['contagion']
listofmaps = calculateSumStats(systemState, metrics, zones)
aguila(listofmaps)

    
