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
    j += 1
#  array = numpy.compress(mask, array)
  print(array)
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
  unique = uniqueid(boolean(spatial(scalar(1))))
  clumps = ifthen(boolean(systemState) == 1, clump(boolean(systemState)))
  numberMap = areadiversity(clumps, spatial(nominal(1))) # doesnt work for test map
  for aStat in listOfSumStats:
    if aStat == 'np': # Number of patches 
      average_nr = cover(areadiversity(clumps, zones), spatial(scalar(0)))  
      listOfMaps.append(average_nr)
    elif aStat == 'pd': # Patch density
      average_nr = cover(areadiversity(clumps, zones), spatial(scalar(0)))
      zone_area = areaarea(zones) # unit? zones are defined as 300, here they are calculated as 30 000
      patch_density = average_nr/zone_area
      listOfMaps.append(patch_density)
    elif aStat == 'mp': # Mean patch size. If patch is in more than one zone it is assigned to one zone only...
      patchSizes = areaarea(clumps)/parameters.getConversionUnit() # each patch has a size assigned
      oneCellPerPatch = pcreq((areamaximum(unique, clumps)), unique) # gets the cell in the right bottom corner of a patch
      patchSizeOneCell = ifthen(oneCellPerPatch, patchSizes)
      averagePatchSize = areaaverage(patchSizeOneCell, zones)
      aguila(averagePatchSize)
      listOfMaps.append(averagePatchSize) 
    elif aStat == 'fd': # Fractal dimension. ### Value of the metric is dependend on the unit used ### Value needs to be higher than e = ~2.71
      scNegative = ifthenelse(boolean(systemState) == 1, boolean(0), boolean(1))
      borders = ifthen(boolean(systemState) == 1, window4total(scalar(scNegative)))
      perimeter = areatotal(borders, nominal(clumps))*sqrt(cellarea())
      patchSizes = areaarea(clumps) # no conversion to km, as we are using meters
      fractalDimension = 2*ln(perimeter)/ln(patchSizes)
      fractalDimensionScalar = cover(fractalDimension, spatial(scalar(0)))
      aguila(fractalDimensionScalar)
      listOfMaps.append(fractalDimensionScalar) 
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
'''
# TEST
""" Testing on the map with one zone: size 30 km x 30 km, with one patch: area 30 km2, perimeter 80 km"""
test_map = os.path.join(os.getcwd(), 'data', 'test_data', 'metric_test_3patches_IE.map')
#aguila(test_map)
systemState = readmap(test_map) == 1 # select urban or predefined pattern
zones_map = os.path.join(inputfolder, 'zones.map')
zones = readmap(zones_map)
# put HERE the name(s) of the metric(s) you want to test
# ['np', 'pd', 'mp', 'fd']
metrics = ['np', 'pd', 'mp', 'fd']
listofmaps = calculateSumStats(systemState, metrics, zones)
#for m in listofmaps:
  #aguila(m)
'''

