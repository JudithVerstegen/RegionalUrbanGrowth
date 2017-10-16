'''
Judith Verstegen, October 2017

This script takes 1) Corine data ('90, '00, '06 and '12) and
2) a map of roads to create all datasets necessary
to run and calibrate PLUC as an urban growth model.

'''

import gdal
import numpy as np
import os
import osr
from pcraster import *
from pcraster.framework import *
from matplotlib import pyplot as plt


##############
### inputs ###
##############

# Directory of Corine land use maps
data_dir = os.path.join('C:\\', 'Users', 'verstege', \
'Documents', 'data')
# Coordinates of case study region
# in ERST 1989 (Corine projection) as [x0, y0, x1, y1]
# current: Madrid
coords = [3127009, 1979498, 3215656, 2064791]
# zone size as a factor of the cell size
zone_size = 100 # 100 x 100 m = 10 000 m = 10x10 km
# for creating observations
realizations = 1
# window size as a factor of the cell size
corr_window_size = 10
omission = 3#52
commission = 3#58

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

def clip(rast, coords):
    '''Clip an opened file and return the clipped file.'''
    # To translate coordinates to raster indices
    gt = rast.GetGeoTransform()
    ul_x, ul_y = world2pixel(gt, coords[0], coords[3])
    lr_x, lr_y = world2pixel(gt, coords[2], coords[1])
    print 'column numbers are', ul_x, ul_y, lr_x, lr_y
    # calculate how many rows and columns the ranges cover
    out_columns = lr_x - ul_x
    # y indices increase from top to bottom!!
    out_rows = lr_y - ul_y
    print 'output raster extent:', out_columns, out_rows
    # Get data from the source raster and write to the new one
    in_band = rast.GetRasterBand(1)
    data = in_band.ReadAsArray(ul_x, ul_y, out_columns, out_rows)
    return data

def clip_and_convert(in_fn, coords, nodata):        
    '''Open, clip and convert dataset to PCRaster map.'''
    rast_data_source = gdal.Open(in_fn)
    # Get georeference info
    geotransform = rast_data_source.GetGeoTransform()
    width = geotransform[1]
    height = geotransform[5]
    print 'Cell size:', width, height
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    print 'x, y:', origin_x, origin_y
    data = clip(rast_data_source, coords)
    themap = numpy2pcr(Nominal, data, nodata)
    ##print data
    return themap

def select_urban(land_use):
    '''Create a Boolean map of all urban land uses from Corine.'''
    urban = scalar(land_use) < 12
    return urban

def simplify_lu_map(amap):
    '''Change Corine map with many classes into simple land use map.'''
    # 1 = urban
    urban = select_urban(amap)
    landuse = nominal(urban)
    # 2 = water and wetlands
    water = pcror(pcrand(scalar(amap) > 34, scalar(amap) < 48),\
                  scalar(amap) == 50)
    landuse = ifthenelse(water, nominal(2), landuse)
    # 3 = nature
    nature = pcror(pcrand(scalar(amap) > 22, scalar(amap) < 35),\
                  scalar(amap) == 49)
    landuse = ifthenelse(nature, nominal(3), landuse)
    # 4 = agriculture
    ag = pcrand(scalar(amap) > 11, scalar(amap) < 23)
    landuse = ifthenelse(ag, nominal(4), landuse)
    return landuse

def omiss_commiss_map(bool_map, randmap, z_omiss, z_commiss):
    # commission error
    to_remove = pcrand(bool_map, randmap >= z_commiss)
    print 'frac remove', \
          float(maptotal(scalar(to_remove))/maptotal(scalar(bool_map)))
    # omission error
    to_add = pcrand(pcrnot(bool_map), randmap >= z_omiss)
    new_map = pcrand(pcrnot(to_remove), pcror(bool_map, to_add))
    return new_map

############
### main ###
############

# 0. clean the two directories (input_data and observations)
files = os.listdir(os.path.join(os.getcwd(), 'input_data'))
for f in files:
    if f not in ['make_demand_manual.xlsx', 'demand_av.tss']:
        os.remove(os.path.join(os.getcwd(), 'input_data', f))
files = os.listdir(os.path.join(os.getcwd(), 'observations'))
for f in files:
    if not os.path.isdir(os.path.join(os.getcwd(), 'observations', f)):
        os.remove(os.path.join(os.getcwd(), 'observations', f))

corine_dir = os.path.join(data_dir, 'Corine')
for a_name in os.listdir(corine_dir):
    # Corine maps are tiffs in folders with same name
    # Except when there is an 'a' behind the version
    if os.path.isdir(os.path.join(corine_dir, a_name)):
        # functions to execute
        # 1. open, clip and convert
        # Path to the old and new raster file
        if a_name[-1] == 'a':
            in_fn = os.path.join(corine_dir, a_name, a_name[0:-1] \
                                 + '.tif')
        else:
            in_fn = os.path.join(corine_dir, a_name, a_name + '.tif')
        print in_fn
        setclone('clone')
        lu = clip_and_convert(in_fn, coords, 48)
        report(lu, 'observations/' + a_name[5:10] + '.map')

        # 2. urban map
        urban = select_urban(lu)
        print a_name[8:10], float(maptotal(scalar(urban)))
        report(urban, 'observations/urb' + a_name[8:10] + '.map')
        ##aguila(urban)
        
        # 3. make simpler initial land use map only for 1990
        if a_name[8:10] == '90':
            simple_lu = simplify_lu_map(lu)
            report(simple_lu, 'input_data/init_lu.map')
            #aguila(simple_lu)
        
# 4. road map outside loop
road_dir = os.path.join(data_dir, 'roads')
in_fn = os.path.join(road_dir, 'roads_spain.tif')
roads = clip_and_convert(in_fn, coords, 255)
nullmask = spatial(nominal(0))
report(cover(roads, nullmask), 'input_data/roads.map')

# 5. other input data sets
# Masks with 0 and 1 for the study area and NoData elsewhere
null_mask = spatial(scalar(0))
report(null_mask, 'input_data/nullmask.map')
one_mask = boolean(null_mask + 1)
report(one_mask, 'input_data/onemask.map')
# Blocks (zones) for the calibration
command = 'resample -r ' + str(zone_size) + ' input_data\onemask.map resamp.map'
os.system(command)
os.system('pcrcalc unique.map = uniqueid(resamp.map)')
os.system('resample unique.map zones.map --clone input_data\onemask.map')
os.system('pcrcalc input_data\zones.map = nominal(zones.map)')
os.remove('zones.map')
os.remove('resamp.map')
os.remove('unique.map')

# 6. calibration and validation blocks 
import covarMatrix
unique = uniqueid(one_mask)
zones = readmap('input_data/zones.map')
samplePoints = pcreq(areamaximum(unique, zones) - 50, unique)
samplePoints = ifthen(samplePoints == 1, boolean(1))
samplePoints = uniqueid(samplePoints)
report(samplePoints, 'input_data/sampPoint.map')
os.system('map2col --unitcell input_data/sampPoint.map input_data/sampPoint.col')
covarMatrix.makeCalibrationMask('input_data/sampPoint.col', zones)

# 7. realizations and their summary statistics
# list of pairs of actual year and time step
if not os.path.exists(os.path.join(os.getcwd(), 'observations', \
                                   'realizations')):
    os.mkdir(os.path.join(os.getcwd(), 'observations', 'realizations'))
for i in range(1, realizations + 1):
    if not os.path.exists(os.path.join(os.getcwd(), 'observations', \
                                       'realizations', str(i))):
        os.mkdir(os.path.join(os.getcwd(), 'observations', 'realizations', str(i)))
    randmap = windowaverage(uniform(1), corr_window_size * celllength())
    arr = pcr2numpy(randmap, np.nan)
    ##plt.hist(arr.flatten())
    ##plt.show()
    z_comm = np.percentile(arr, 100 - commission)
    ##print z_comm
    for year in [('90', 0), ('00', 10), ('06', 16), ('12', 22)]:
        amap = readmap('observations/urb' + year[0] + '.map')
        # omission needs to be corrected for class occurance
        x = float(maptotal(scalar(amap))/maptotal(scalar(pcrnot(amap))))
        ##print x
        z_om = np.percentile(arr, 100 - (omission * x))
        ##print z_om
        new_map = omiss_commiss_map(amap, randmap, z_om, z_comm)
        report(new_map, generateNameT('observations/realizations/' + \
                                     str(i) + '/urb', year[1]))
        print year[0], float(maptotal(scalar(new_map)))
        # for calibration AND validation
        for cal in (True, False):
            listOfSumStats = covarMatrix.calculateSumStats(new_map, \
                                                            ['av', 'nr', 'ls'],\
                                                            zones, cal)
            observedAverageMap = listOfSumStats[0]
            observedNumberMap = listOfSumStats[1] 
            observedPatchMap = listOfSumStats[2]

            if cal:
                report(observedAverageMap, \
                       generateNameT('observations/realizations/' + \
                                     str(i) + '/av_c', year[1]))
                report(observedNumberMap, \
                       generateNameT('observations/realizations/' + \
                                     str(i) + '/nr_c', year[1]))
                report(observedPatchMap, \
                       generateNameT('observations/realizations/' + \
                                     str(i) + '/ls_c', year[1]))
            else: # val
                report(observedAverageMap, \
                       generateNameT('observations/realizations/' + \
                                     str(i) + '/av_v', year[1]))
                report(observedNumberMap, \
                       generateNameT('observations/realizations/' + \
                                     str(i) + '/nr_v', year[1]))
                report(observedPatchMap, \
                       generateNameT('observations/realizations/' + \
                                     str(i) + '/ls_v', year[1]))            
        
# 8. covar matrices
sample_numbers=range(1, realizations+1, 1)
time_steps = [0, 10, 16, 22]
covarMatrix.mcCovarMatrix(["av", "nr", "ls"],sample_numbers, \
          time_steps, ['input_data/sampPointAvSelection.col', \
                       'input_data/sampPointNrSelection.col',\
                     'input_data/sampPointNrSelection.col'], "covar","corr")
names = ['av', 'nr', 'ls']
percentiles = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
mcaveragevariance(names, sample_numbers, time_steps)
