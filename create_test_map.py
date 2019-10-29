'''
Judith Verstegen, May 2019

This script creates test map with a predefined pattern to be used for the metrics testing.

'''

import gdal # version 2.3.3 for Python 3.6
import numpy as np
import os
import osr
import ogr
import parameters
from pcraster import *
from pcraster.framework import *
from matplotlib import pyplot as plt



##############
### inputs ###
##############

# Directory of Corine land use maps and other input
data_dir = os.path.join(os.getcwd(), 'data')

# Coordinates of case study region
# in ERST 1989 (Corine projection) as [x0, y0, x1, y1]
country = parameters.getCountryName()
coords_dict = {
    'IE':[3167978,3406127,3327978,3566127],
    'IT':[4172280,2403670,4332280,2563670],
    'PL':[5002510,3212710,5162510,3372710]
}

# current: IE
coords = coords_dict[country] 

#################
### functions ###
#################


def world2pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate (Cookbook)
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)

def getRowsCols(rast, coords):
    """Calculate nr of rows and colums of output file"""
    # To translate coordinates to raster indices
    gt = rast.GetGeoTransform()
    ul_x, ul_y = world2pixel(gt, coords[0], coords[3])
    lr_x, lr_y = world2pixel(gt, coords[2], coords[1])
    print('column numbers are', ul_x, ul_y, lr_x, lr_y)
    # calculate how many rows and columns the ranges cover
    out_columns = lr_x - ul_x
    # y indices increase from top to bottom!!
    out_rows = lr_y - ul_y
    return out_rows, out_columns, ul_x, ul_y

def clip(rast, coords):
    '''Clip an opened file and return the clipped file.'''
    out_rows, out_columns, ul_x, ul_y = getRowsCols(rast, coords)
    print('output raster extent:', out_columns, out_rows)
    # Get data from the source raster and write to the new one
    in_band = rast.GetRasterBand(1)
    data = in_band.ReadAsArray(ul_x, ul_y, out_columns, out_rows)
    return data

def clip_and_convert(in_fn, coords, nodata, datatype):        
    '''Open, clip and convert dataset to PCRaster map.'''
    '''datatype: Boolean, Nominal, Ordinal, Scalar, Directional or Ldd'''
    rast_data_source = gdal.Open(in_fn)
    # Get georeference info
    geotransform = rast_data_source.GetGeoTransform()
    width = geotransform[1]
    height = geotransform[5]
    print('Cell size:', width, height)
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    print('x, y:', origin_x, origin_y)
    data = clip(rast_data_source, coords)
    print('Nominal?')
    themap = numpy2pcr(datatype, data, nodata)
    print('Nominal')
    ##print data
    del rast_data_source
    return themap

def metric_test_map(amap):
    test_map = nominal(amap)
    return test_map

############
### main ###
############

# Select the dir for the temporal working files
test_dir = os.path.join(data_dir, 'test_data')
# Select clone map. Clone map needs to exist.
setclone('clone')

print('-------------------- Create test map --------------------')
test_patch = os.path.join(test_dir, 'PL_test_3patches_recl_clip1.tif')
test_map = clip_and_convert(test_patch, coords, 0, Nominal)
nullmask = spatial(nominal(0))
t_map=cover(nominal(test_map), nullmask)
aguila(t_map)
report(t_map, os.path.join(test_dir, 'metric_test_3patches_PL.map'))
