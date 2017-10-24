# -*- coding: utf-8 -*-
import math
import os
import shutil
import string
import numpy
import numpy.ma
from pcraster import *
from pcraster.framework import * 
import sys
#sys.path.insert(0, os.path.abspath("C:\Program Files\pcraster-4.1.0_x86-64\python\pcraster\framework"))
#sftp://judith@fleet.geo.uu.nl/opt/PCRaster-3.0.0-beta-100504/Python/PCRaster/Framework
#sys.path.insert("opt/PCRaster-3.0.0-beta-100504/Python/PCRaster/Framework")
#from frameworkBase import generateNameS, generateNameT, generateNameST
#import generalfunctions, regression




def average(name, sampleNumbers):
  """
  Calculates the average value of each cell.

  name
    Name of the scalar raster for which each sample has a realization.

  sampleNumbers
    List of numbers of samples to aggregate.

  Returns a raster with average values.
  """
  sum = scalar(0)
  count = scalar(0)
  for sample in sampleNumbers:
    filename = generateNameS(name, sample)
    raster   = scalar(readmap(filename))
    sum      = sum + raster
    count    = ifthen(defined(raster), count + 1)
  return sum / count



def variance(name, sampleNumbers):
  """
  Calculates the variance of each cell.

  name
    Name of the scalar raster for which each sample has a realization.

  sampleNumbers
    List of numbers of samples to aggregate.

  Returns a raster with variances.
  """
  sumOfSquaredValues, sumOfValues, count = scalar(0), scalar(0), scalar(0)
  for sample in sampleNumbers:
    filename           = generateNameS(name, sample)
    raster             = scalar(readmap(filename))
    sumOfSquaredValues = sumOfSquaredValues + raster ** 2
    sumOfValues        = sumOfValues + raster
    count              = ifthen(defined(raster), count + 1)
  return (count * sumOfSquaredValues - sumOfValues ** 2) / (count * (count - 1))



def staticInput(timeSteps):
  return len(timeSteps) == 1 and timeSteps[0] == 0



def deterministicInput(sampleNumbers):
  return len(sampleNumbers) == 1 and sampleNumbers[0] == 0



def sampleMin(name, sampleNumbers):
  """
  Calculates the minimum value of each cell.

  name
    Name of the scalar raster for which each sample has a realization.

  sampleNumbers
    List of numbers of samples to aggregate.

  Returns a raster with minimum values.
  """
  minimum = scalar(1e31)
  for sample in sampleNumbers:
    filename = generateNameS(name, sample)
    raster   = scalar(readmap(filename))
    minimum      = ifthenelse(pcrlt(raster,minimum),raster,minimum)
  return minimum



def sampleMax(name, sampleNumbers):
  """
  Calculates the maximum value of each cell.

  name
    Name of the scalar raster for which each sample has a realization.

  sampleNumbers
    List of numbers of samples to aggregate.

  Returns a raster with maximum values.
  """
  maximum = scalar(-1e31)
  for sample in sampleNumbers:
    filename = generateNameS(name, sample)
    raster   = scalar(readmap(filename))
    maximum      = ifthenelse(pcrgt(raster,maximum),raster,maximum)
  return maximum


def mcaveragevariance(names,sampleNumbers, timeSteps):
  if staticInput(timeSteps):
    for name in names:
      mean=average(name + '.map', sampleNumbers)
      var=variance(name + '.map', sampleNumbers)
      minimum=sampleMin(name + '.map', sampleNumbers)
      maximum=sampleMax(name + '.map', sampleNumbers)
      #std=stddev(name + '.map', sampleNumbers)
      report(mean, name + '-ave.map')
      report(var, name + '-var.map')
      report(minimum, name + '-min.map')
      report(maximum, name + '-max.map')
      report(sqrt(var)/mean, name + '-err.map')
  else:
    nrSamples=scalar(len(sampleNumbers))
    for name in names:
      for step in timeSteps:
        var=variance(generateNameT(name,step),sampleNumbers)
        mean=average(generateNameT(name,step), sampleNumbers)
        report(mean, generateNameT(name + '-ave', step))
        report(var, generateNameT(name + '-var', step))
        report(sqrt(var)/mean, generateNameT(name + '-err', step))

# test
##names = ['sSc']
##sampleNumbers = [1,2,3,4,5]
##timeSteps = [1,2,3,4,5,6,7,8,10]#,12,14,16,18,20] #self.timeSteps()
##mcaveragevariance(names, sampleNumbers, timeSteps)
