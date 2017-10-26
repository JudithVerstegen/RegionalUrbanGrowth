import string
import os, shutil, numpy
#import generalfunctions
import matplotlib.pyplot as plt
from pcraster import *
from pcraster.framework import *
import parameters

def map2Array(aMap, rowColFile):
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
##  mask = numpy.zeros((1, len(samplePoints))).astype(numpy.bool_)
##  array = numpy.zeros((1, len(samplePoints))).astype(numpy.float32)
  mask = numpy.zeros(len(samplePoints)).astype(numpy.bool_)
  array = numpy.zeros(len(samplePoints)).astype(numpy.float32)
  j = 0
  for point in samplePoints:
    attributes = string.split(point)
##      print attributes
    row = int(round(float(attributes[1])))
    col = int(round(float(attributes[0])))
##    array[[0],[j]], mask[[0],[j]] = readFieldCell(aMap, row, col)
    array[j], mask[j] = readFieldCell(aMap, row, col)
    j += 1
#  array = numpy.compress(mask, array)
#  print array
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
      col = int(round(float(attributes[1])))
      row = int(round(float(attributes[0])))
      array[[j], [i]], mask[[j], [i]] = cellvalue(amap, row, col)
      j += 1
    i += 1
#  array = numpy.compress(mask, array)
  print 'array', array
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

def covarMatrix(names,sampleNumbers,rowColFiles,covarMatrixName, \
                corrMatrixName, base=None):
  dataMatrix = selectSArrayMultipleRasters(names,sampleNumbers,rowColFiles,\
                                           base)
  numpy.savetxt('dataMatrix',dataMatrix)
  covarMatrix = numpy.cov(dataMatrix)
  print 'covarMatrix', covarMatrix.shape[0]
  corrMatrix = numpy.corrcoef(dataMatrix)
  if base is not None:
    covarMatrixName = os.path.join(base, covarMatrixName)
    corrMatrixName = os.path.join(base, corrMatrixName)
  # This part was because Derek uses only 1 obs/timestep, which makes 1x1 matrix
  if len(names) == 0:#1:
    covarMatrixDifferentType=numpy.array([float(covarMatrix)])
    corrMatrixDifferentType=numpy.array([float(corrMatrix)])
    numpy.savetxt(covarMatrixName,covarMatrixDifferentType)
    numpy.savetxt(corrMatrixName,corrMatrixDifferentType,fmt="%3.2f")
  else:
    numpy.savetxt(covarMatrixName,covarMatrix)
    numpy.savetxt(corrMatrixName,corrMatrix,fmt="%3.2f")

def mcCovarMatrix(names,sampleNumbers,timeSteps,rowColFile,\
                  covarMatrixBaseName,corrMatrixBaseName, base=None):
  for step in timeSteps:
    namesForTimestep=[]
    for name in names:
      namesForTimestep.append(generateNameT(name,step)) 
    covarMatrixName=generateNameT(covarMatrixBaseName,step)
    corrMatrixName=generateNameT(corrMatrixBaseName,step)
    covarMatrix(namesForTimestep, sampleNumbers, rowColFile, \
                covarMatrixName,corrMatrixName, base)

def calculateSumStats(systemState, listOfSumStats, zones, validation=False):
  """Return a list of sum stat maps for the sum stat"""
  listOfMaps = []
##  mask = readmap('input_data/zones_selection')
##  if validation == True: mask = readmap('input_data/zones_validation')
##  systemState = ifthen(mask, systemState)
  unique = uniqueid(boolean(spatial(scalar(1))))
  clumps = ifthen(boolean(systemState) == 1, clump(boolean(systemState)))
  numberMap = areadiversity(clumps, spatial(nominal(1)))
  for aStat in listOfSumStats:
    if aStat == 'av':
      averageMap = areaaverage(scalar(systemState), zones)
      listOfMaps.append(averageMap)
    elif aStat == 'nr':
      average_nr = cover(areadiversity(clumps, zones), spatial(scalar(0)))
      listOfMaps.append(average_nr)
    elif aStat == 'ps':
      patchSizes = areaarea(clumps)/parameters.getConversionUnit()
      oneCellPerPatch = pcreq(areamaximum(unique, clumps), unique)
      patchSizeOneCell = ifthen(oneCellPerPatch, patchSizes)
      averagePatchSize = maptotal(patchSizeOneCell)/numberMap
      listOfMaps.append(averagePatchSize)
    elif aStat == 'ls':
      scNegative = ifthenelse(boolean(systemState) == 1, boolean(0), boolean(1))
      borders = ifthen(boolean(systemState) == 1, \
                       window4total(scalar(scNegative)))
      totalPerimeter= maptotal(borders)
      smallestPerimeter = sqrt(maptotal(scalar(systemState))) * 4
      # non-zero number for division
      if smallestPerimeter < 0.0001:
        lsi = 0
      else:
        lsi = totalPerimeter/smallestPerimeter
      listOfMaps.append(lsi)
    else:
      print 'ERRRRRRRRRRRROR, unknown sum stat'
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
  print selection
  # make a new row col file for the blocks
  newTextfile1 = open('input_data/sampPointAvSelection.col', 'w')
  newTextfile2 = open('input_data/sampPointNrSelection.col', 'w')
  lookuptable1 = open('input_data/lookupTable_cal.tbl', 'w')
  newTextfile3 = open('input_data/sampPointAvValidation.col', 'w')
  newTextfile4 = open('input_data/sampPointNrValidation.col', 'w')
  lookuptable2 = open('input_data/lookupTable_val.tbl', 'w')
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
  blocksTrue = lookupboolean('input_data/lookupTable_cal.tbl', zoneMap)
  report(blocksTrue, 'input_data/zones_selection.map')
  blocksTrue = lookupboolean('input_data/lookupTable_val.tbl', zoneMap)
  report(blocksTrue, 'input_data/zones_validation.map')

### test
###createTimeseries("jan", [1,2,3],[1,2], 1, 1,"test.pdf")
###createTimeseriesConfInt("1/jan", "1/piet","2/jan", [1,2,3,4,5,6,7,8,9], 1, 1,"test.pdf")
##
  
##setclone('input_data/nullmask.map')
##sampleNumbers=range(3,51,1)
##timeSteps=range(2,6,1)
##mcCovarMatrix(['av'],sampleNumbers,timeSteps,\
##                           'input_data/sampPoint.col',"input_data/covar",\
##              "input_data/corr")
##array = mySelectSArray('nr000000.002', range(1,51), \
##                       'input_data/sampPoint.col')
##print '\n'
##matrix2 = map2Array('test', 'input_data/sampPoint.col')
##zones = readmap('zones')
##makeCalibrationMask('input_data/sampPointAv.col', zones)
