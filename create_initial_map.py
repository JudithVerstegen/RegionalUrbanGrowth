'''
Judith Verstegen, May 2019

This script takes 1) Corine data ('90, '00, '06, '12 and '18) and
2) a map of roads to create all datasets necessary
to generate inputs and calibration data for an urban growth model.

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

# Set the country bounding box
coords = coords_dict[country] 

# zone size as a factor of the cell size
numberOfZones = parameters.getNumberOfZones()
zone_size = int(np.sqrt(numberOfZones)*100) # 100 m (cell size) -> zone_size = 40 km. Test map, dividing area into 16 zones. ## The transform_save_data.py script didn't work for 4 zones
# for creating observations
realizations = 20
# window size as a factor of the cell size
corr_window_size = 50
omission = 10#52
metric_names = parameters.getSumStats()

#################
### functions ###
#################

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

def makeclone(in_fn, coords):
    """Uses gdal and pcraster to automatically create a clone file."""
    # remove existing clone if present
    if os.path.exists('./clone.map'): os.remove('./clone.map')
    # open the raster
    rast = gdal.Open(in_fn)
    out_rows, out_columns, ul_x, ul_y = getRowsCols(rast, coords)
    # Make the clone with the following inputs
    # -s for not invoking the menu
    # -R nr of rows
    # -C nr of columns
    # -N data type Nominal
    # -P y coordinates increase bottom to top
    # -x x-coordinate of upper left corner
    # -y y-coordinate of upper left corner
    # -l cell length, set to 100 m (same as Corine)
    strings = ['mapattr -s', ' -R ' + str(out_rows), \
               ' -C ' + str(out_columns), ' -N ',  '-P yb2t', \
               ' -x ' + str(coords[0]), ' -y ' + str(coords[3]), \
               ' -l 100 clone.map']       
    command = "".join(strings)
    print(command)
    os.system(command)

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
    themap = numpy2pcr(datatype, data, nodata)
    ##print data
    del rast_data_source
    return themap

def select_urban(land_use):
    '''Create a Boolean map of all urban land uses from Corine.Without 122 (roads) and 124 (airports)'''
    urban = pcrand(scalar(land_use) < 200, pcrand(scalar(land_use) > 0,pcrand(pcrnot(scalar(land_use) == 122), pcrnot(scalar(land_use) == 124))))
    #aguila(urban)
    return urban
    
def simplify_lu_map(amap):
    '''Change Corine map with many classes into simple land use map.'''
    # 1 = urban
    urban = select_urban(amap)
    landuse = nominal(urban)
    # 2 = water AND ROADS
    water = pcror(pcror(pcrand(scalar(amap) > 500, scalar(amap) < 900), scalar(amap) == 122), scalar(amap) == 124)
    landuse = ifthenelse(water, nominal(2), landuse)
    # 3 = nature
    nature = pcrand(scalar(amap) > 300, scalar(amap) < 500)
    landuse = ifthenelse(nature, nominal(3), landuse)
    # 4 = agriculture
    ag = pcrand(scalar(amap) > 200, scalar(amap) < 300)
    landuse = ifthenelse(ag, nominal(4), landuse)
    # 5 = NODATA # find nodata in pcraster
    nodata = (landuse==0)
    landuse = ifthenelse(nodata, nominal(5), landuse)

    landuse = ifthen(scalar(landuse)<5, landuse)
    return landuse

def omiss_commiss_map(prev, bool_map, randmap, omiss, simple_lu):
    # commission error
    arr = pcr2numpy(randmap, np.nan)
    z_comm = np.percentile(arr, 100 - omiss)
    to_remove = pcrand(bool_map, randmap >= z_comm)
    ##print 'frac remove', \
    float(maptotal(scalar(to_remove))/maptotal(scalar(bool_map)))
    cells = float(maptotal(scalar(to_remove)))
    ##print 'cells to remove', cells

    # omission needs to be corrected for class occurance
    where_can = pcrand(pcrand(pcrnot(bool_map), pcrnot(prev)), \
                pcrne(simple_lu, 2))
    x = float (cells / maptotal(scalar(where_can)))
    ##print 'frac add', x
    z_om = np.percentile(arr, 100 - (100 * x))
    # omission error
    to_add = pcrand(where_can, randmap >= z_om)
    cells = float(maptotal(scalar(to_add)))
    ##print 'cells to add', cells

    ##new_map = pcrand(pcrnot(to_remove), pcror(bool_map, to_add))
    return to_remove, to_add

def reproject(in_fn, out_fn, in_rast, data_type):
    # data_type: ['point','polyline','polygon']
    
    print('Reprojecting shapefile...')
    # Open the raster
    rast_data_source = gdal.Open(in_rast)

    # Get metadata (not required)
    print('nr of bands:', rast_data_source.RasterCount)
    cols = rast_data_source.RasterXSize
    rows = rast_data_source.RasterYSize
    print('extent:', cols, rows)

    # Get georeference info (not required)
    geotransform = rast_data_source.GetGeoTransform()
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    print('cell size:', pixelWidth, pixelHeight)
    originX = geotransform[0]
    originY = geotransform[3]
    print('x, y:', originX, originY)
    
    rast_spatial_ref = rast_data_source.GetProjection()
    
    # Get the correct driver
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # 0 means read-only. 1 means writeable.
    vect_data_source = driver.Open(in_fn, 0) 

    # Check to see if shapefile is found.
    if vect_data_source is None:
        print('Could not open %s' % (in_fn))

    # Get the Layer class object
    layer = vect_data_source.GetLayer(0)
    # Get reference system info
    vect_spatial_ref = layer.GetSpatialRef()
    #print('vector spatial ref is', vect_spatial_ref)

    # create osr object of raster spatial ref info
    sr = osr.SpatialReference(rast_spatial_ref)
    transform = osr.CoordinateTransformation(vect_spatial_ref, sr)

    # Delete if output file already exists
    # We can use the same driver
    if os.path.exists(out_fn):
        print('exists, deleting')
        driver.DeleteDataSource(out_fn)
    out_ds = driver.CreateDataSource(out_fn)
    if out_ds is None:
        print('Could not create %s' % (out_fn))

    # Create the shapefile layer WITH THE SR
    if data_type == 'point':
        data_arg = ogr.wkbPoint
    elif data_type == 'polyline':
        data_arg = ogr.wkbLineString
    elif data_type == 'polygon':
        data_arg = ogr.wkbLinearRing
        
    out_lyr = out_ds.CreateLayer('reprojected', sr, 
                                 data_arg)

    out_lyr.CreateFields(layer.schema)
    out_defn = out_lyr.GetLayerDefn()
    out_feat = ogr.Feature(out_defn)
    # Loop over all features and change their spatial ref
    for in_feat in layer:
        geom = in_feat.geometry()
        geom.Transform(transform)
        out_feat.SetGeometry(geom)
        # Make sure to also include the attributes in the new file
        for i in range(in_feat.GetFieldCount()):
            value = in_feat.GetField(i)
            out_feat.SetField(i, value)
        out_lyr.CreateFeature(out_feat)

    del out_ds
    print('Reprojected.')

    return out_feat

def rasterize(InputVector, OutputImage, RefImage):
    print('Rasterizing shapefile...')
    gdalformat = 'GTiff'
    datatype = gdal.GDT_Byte
    burnVal = 1 #value for the output image pixels

    # Get projection info from reference image
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    Shapefile = ogr.Open(InputVector)
    Shapefile_layer = Shapefile.GetLayer()

    # Chceck if raster exists. If yes, delete.
    if os.path.exists(OutputImage):
        print('Raster exists, deleting')
        os.remove(OutputImage) 

    # Rasterise
    Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype, options=['COMPRESS=DEFLATE'])
    print('New raster created')
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform()) 

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(255)
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

    # Close datasets
    Band = None
    Output = None
    Image = None
    Shapefile = None

    # Build image overviews
    subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE "+OutputImage+" 2 4 8 16 32 64", shell=True)
    print("Rasterized.")

def create_filtered_shapefile(in_shapefile, country, out_dir, out_name, filter_query):
    ### Script for selecting train stations from OSM transport data.
    ### Data was downlowaded for Ireland, Italy and Poland.
    ### Data is in folders with names corresponding to the names of the countries
    ### train station attributes: railway=halt and railway=station
    
    print('Filtering shapefile...')
    # Get the correct driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # 0 means read-only. 1 means writeable.
    data_source = driver.Open(in_shapefile,0)
    
    # Check to see if shapefile is found.
    if data_source is None:
        print('Could not open %s' % (in_shapefile))
    
    # get the Layer class object
    input_layer = data_source.GetLayer(0)
    
    # Apply a filter
    input_layer.SetAttributeFilter(filter_query)
    
    # Copy Filtered Layer and Output File
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Check if output data exists. If yes, delete.
    if os.path.exists(os.path.join(out_dir, out_name)):
        print('Shapefile exists, deleting')
        os.remove(os.path.join(out_dir, out_name))
    out_ds = driver.CreateDataSource(out_dir)
    print('Filtered')
    out_layer = out_ds.CopyLayer(input_layer, out_name)
    
    del input_layer, out_layer, out_ds

def reproject_resample_tif(in_raster, out_raster, ref_raster):
    ''' Reprojects the inout raster to match the CRS and cell size of the reference raster.'''
    ''' Resampling method: Nearest Neighbour.'''
    print('Reprojecting and resampling tif...')

    theinput = gdal.Open(in_raster, gdal.GA_ReadOnly)
    inputProj = theinput.GetProjection()
    inputTrans = theinput.GetGeoTransform()

    reference = gdal.Open(ref_raster, gdal.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    bandreference = reference.GetRasterBand(1)    
    x = reference.RasterXSize 
    y = reference.RasterYSize
    
    # Chceck if raster exists. If yes, delete.
    if os.path.exists(out_raster):
        print('Raster exists, deleting')
        os.remove(out_raster)
        
    driver= gdal.GetDriverByName('GTiff')
    output = driver.Create(out_raster,x,y,1,bandreference.DataType)
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)

    gdal.ReprojectImage(theinput,output,inputProj,referenceProj,gdal.GRA_NearestNeighbour)
    print('Tif reprojected and resampled')

    del theinput, output, reference

############
### main ###
############

# 0. clean the two directories (input_data and observations)
# Folders input_data and observations have to exist
country_dir = os.path.join(os.getcwd(), 'input_data', str(country))
if not os.path.isdir(country_dir):
    os.mkdir(country_dir)
if not os.path.isdir(os.path.join(os.getcwd(), 'observations', str(country))):
    os.mkdir(os.path.join(os.getcwd(), 'observations', str(country)))

files = os.listdir(country_dir)
for f in files:
    if f not in ['make_demand_manual.xlsx', 'demand.tss']:
        os.remove(os.path.join(country_dir, f))
files = os.listdir(os.path.join(os.getcwd(), 'observations', str(country)))
for f in files:
    if not os.path.isdir(os.path.join(os.getcwd(), 'observations', \
                                      str(country), f)):
        os.remove(os.path.join(os.getcwd(), 'observations', str(country), f))

# create the clone map
corine_dir = os.path.join(data_dir, 'Corine')
names = os.listdir(corine_dir)
# hereto we need one Corine raster. It does not matter which one
print(os.path.join(corine_dir, names[1]))
if os.path.isdir(os.path.join(corine_dir, names[1])):
    print('here')
    in_fn = os.path.join(corine_dir, names[1], names[1] + '.tif')
    makeclone(in_fn, coords)

# Corine maps
print('----------------------- Urban maps -----------------------')
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
        print(in_fn)
        setclone('clone')
        lu = clip_and_convert(in_fn, coords, 999, Nominal)
        report(lu, os.path.join('observations', country, a_name[13:15] + '.map'))

        # 2. urban map
        urban = select_urban(lu)
        print(a_name[13:15], float(maptotal(scalar(urban))))
        report(urban, os.path.join('observations', country, \
                                   'urb' + a_name[13:15] + '.map'))
        
        # 3. make simpler initial land use map only for 1990
        if a_name[13:15] == '90':
            simple_lu = simplify_lu_map(lu)
            report(simple_lu, os.path.join(country_dir, 'init_lu.map'))

# Select the dir for the temporal working files
temp_dir = os.path.join(data_dir, 'temporal_data')

print('-------------------- Reference raster --------------------')
# Select the 1990 Corine raster as the reference raster for further actions
raster_name = os.listdir(corine_dir)[0]
print('Reference raster name ' + raster_name)
ref_raster = os.path.join(corine_dir, raster_name, raster_name + '.tif')
    
# 4. road map outside loop
print('------------------------- Roads -------------------------')
# Road dataset will be reprojected and rasterized and saved into 'raster' folder inside the road_dir.
# 'raster' folder needs to exist.
road_dir = os.path.join(data_dir, 'roads')

# Reproject the input vector data using the raster as the reference layer
# Save the reprojected file in the input_data folder and remove later
in_shp = os.path.join(road_dir, 'roads.shp')
out_fn = os.path.join(data_dir, 'temporal_data', 'roads_reprojected.shp')
reproject(in_shp, out_fn, ref_raster, 'polyline')
# Rasterize the reprojected shapefile
out_raster = os.path.join(data_dir, 'temporal_data', 'roads_raster.tif')
rasterize(out_fn, out_raster, ref_raster)
roads = clip_and_convert(out_raster, coords, 255, Nominal)
nullmask = spatial(nominal(0))

report(cover(roads, nullmask), os.path.join(country_dir, 'roads.map'))
# Remove the working files
road_files = os.listdir(temp_dir)
for f in road_files:
    os.remove(os.path.join(temp_dir, f))
print('Roads created.')

# 5. train station map outside loop
print('-------------------- Train stations --------------------')
# Train station dataset will be reprojected and rasterized and saved into 'raster' folder inside the railways_dir.
# 'raster' folder needs to exist.
railway_dir = os.path.join(data_dir, 'railways')

print('Creating train stations map in: ', country,'...')

### 1. Select the train stations
# Select the input and output shapefile dir and name
in_fn = os.path.join(railway_dir, country, 'gis_osm_transport_free_1.shp')
f_name = 'stations_' + country
out_dir = os.path.join(data_dir, 'temporal_data')
out_name = 'stations_' + country
# Filter by our query
query_str = "fclass = 'railway_station' OR fclass = 'railway_halt'"
create_filtered_shapefile(in_fn, country, out_dir, out_name, query_str)
print(country, ': Filtered shapefile created.')

### 2. Reproject the shapefiles
in_shp = os.path.join(data_dir, 'temporal_data', out_name + '.shp')
out_shp = os.path.join(data_dir, 'temporal_data', out_name + '_reprojected.shp')
reproject(in_shp, out_shp, ref_raster, 'point')

### 3. Rasterize the reprojected shapefile
f_dir = os.path.join(railway_dir, country)
##raster_dir = os.path.join(f_dir, 'raster')
out_raster = os.path.join(data_dir, 'temporal_data', 'stations_' + country + '.tif')
rasterize(out_shp, out_raster, ref_raster)

### 4. Create the map files
stations = clip_and_convert(out_raster, coords, 255, Nominal)
nullmask = spatial(nominal(0))
report(cover(stations, nullmask), os.path.join(country_dir, 'train_stations.map'))
    
### 5. Remove the working files
rail_files = os.listdir(temp_dir)
for f in rail_files:
    os.remove(os.path.join(temp_dir, f))
print('Train stations created.')

# 6. no-go areas map outside loop  <- to be merged into one no-go map
# Protected areas from NATURA2000 database and areas with the slope > 30 degrees are excluded
print('---------------------- No-go map ----------------------')

print('---------- 1. Protected areas ----------')
# Excluded areas will be rasterized and saved into 'raster' folder inside the protected_dir.
# 'raster' folder needs to exist.
protected_dir = os.path.join(data_dir, 'NATURA2000')

### 1. Rasterize the projected area shapefile
# Select the input and output dir and name
in_fn = os.path.join(protected_dir, 'Natura2000_end2018_epsg3035.shp')
##raster_dir = os.path.join(protected_dir, 'raster')
out_raster = os.path.join(data_dir, 'temporal_data','protected.tif')
rasterize(in_fn, out_raster, ref_raster)

### 2. Create the map file
protected = clip_and_convert(out_raster, coords, 255, Nominal)
report(cover(protected, nullmask), os.path.join(country_dir, 'NATURA2000.map'))
print('Protected areas created.')

print('----------- 2. Steep areas -----------')
# Opend the DEM
dem_names = {
    'IT': 'eu_dem_v11_E40N20',
    'IE': 'eu_dem_v11_E30N30',
    'PL': 'eu_dem_v11_E50N30' 
    }
dem_dir = os.path.join(data_dir, 'DEM', country, dem_names[country] + '.TIF')
dem_repr = os.path.join(temp_dir, 'reprojected_dem_' + country + '.tif')
# Reproject and resample DEM tif to match the CLC dataset
reproject_resample_tif(dem_dir, dem_repr, ref_raster)
# Clip and convert
dem = clip_and_convert(dem_repr, coords, 255, Scalar)
nullmask = spatial(scalar(0))
report(cover(dem, nullmask), os.path.join(country_dir, 'dem.map'))
print('DEM map created.')
# Calculate the slope
slope = slope(dem)
report(cover(slope, nullmask), os.path.join(country_dir, 'slope.map'))
print('Slope map created.')
# Remove the working files
dem_files = os.listdir(temp_dir)
for f in dem_files:
    os.remove(os.path.join(temp_dir, f))


print('------------ 3. No-go map ------------')
# Combine the maps into the no-go map
# NATURA2000 and slope >30 degrees (around 58%) are excluded
nogo = pcror(boolean(protected),slope>=0.58)
report(nogo, os.path.join(country_dir, 'nogo.map')) # SHOULD I USE COVER() FUNCTION?
print('nogo map created.')

# 7. other input data sets
# Masks with 0 and 1 for the study area and NoData elsewhere
null_mask = spatial(scalar(0))
report(null_mask, os.path.join(country_dir, 'nullmask.map'))
one_mask = boolean(null_mask + 1)
report(one_mask, os.path.join(country_dir, 'onemask.map'))
# Blocks (zones) for the calibration
command = 'resample -r ' + str(zone_size) + ' ' + \
          os.path.join(country_dir, 'onemask.map') + ' resamp.map'
os.system(command)
os.system('pcrcalc unique.map = uniqueid(resamp.map)')
command = 'resample unique.map zones.map --clone ' + \
          os.path.join(country_dir, 'onemask.map')
os.system(command)
os.system('pcrcalc ' + os.path.join(country_dir, 'zones.map') + \
          ' = nominal(zones.map)')
os.remove('zones.map')
os.remove('resamp.map')
os.remove('unique.map')

# 8. blocks 
import metrics
unique = uniqueid(one_mask)
zones = readmap(os.path.join(country_dir, 'zones.map'))

### remove zones with zero urban in 2000
##urb2000 = readmap('observations\urb00')
##av = areaaverage(scalar(urb2000), zones)
##zones = ifthen(av > 0, zones)
##report(zones, 'input_data\zones.map')

samplePoints = pcreq(areaminimum(unique, zones), unique)
samplePoints = ifthen(samplePoints == 1, boolean(1))
samplePoints = uniqueid(samplePoints)
report(samplePoints, os.path.join(country_dir, 'sampPoint.map'))
command = 'map2col --unitcell ' + os.path.join(country_dir, 'sampPoint.map') + \
          ' ' + os.path.join(country_dir, 'sampPoint.col')
os.system(command)

# 9. summary statistics with no stochasticity
print('---------------------- Statistics ----------------------')
print('------------ CLC area ------------')
area = {}

for year in ['90','00','06','12','18']:
    amap = readmap(os.path.join(os.getcwd(), 'observations', \
                                country, 'urb' + year + '.map'))
    # collect total area data for demand
    area[year] = float(maptotal(scalar(amap)))
print('Area from CLC:')
print(area)
print('')

# 10. realizations and their summary statistics
print('------------ Realizations ------------')
# list of pairs of actual year and time step
if not os.path.exists(os.path.join(os.getcwd(), 'observations', \
                                   country, 'realizations')):
    os.mkdir(os.path.join(os.getcwd(), 'observations', country, 'realizations'))

avs = {}
mins = {}
maxs = {}
for i in range(1, realizations + 1):
    print(i)
    # make directories for the realizations
    if not os.path.exists(os.path.join(os.getcwd(), 'observations', country, \
                                       'realizations', str(i))):
        os.mkdir(os.path.join(os.getcwd(), 'observations', country, \
                              'realizations', str(i)))
    # map with random numbers but with correlation by moving window
    randmap = windowaverage(uniform(1), corr_window_size * celllength())
    base = os.path.join('observations', country, 'realizations')
    prev = None
    # Year numbers are adjusted to the time step numbers
    for year in [('90', 1), ('00', 11), ('06', 17), ('12', 23), ('18', 29)]: 
        amap = readmap(os.path.join(os.getcwd(), 'observations', \
                                    country, 'urb' + year[0] + '.map'))
        # change some of the NEW urban cells, not the existing ones
        if prev is not None:
            diff = pcrand(amap, pcrnot(prev))
            ##aguila(diff)
            to_remove, to_add = omiss_commiss_map(prev, diff, randmap, \
                                                  omission, simple_lu)
            new_map = pcrand(pcrnot(to_remove), pcror(amap, to_add))
            ##aguila(new_map)
        else:
            new_map = amap

        # TO_DO collect total area data for demand
        cells = float(maptotal(scalar(new_map)))
        if i == 1:
            mins[year[0]] = cells
            maxs[year[0]] = cells
            avs[year[0]] = float(maptotal(scalar(amap)))
        else:
            if cells < mins[year[0]]: mins[year[0]] = cells
            if cells > maxs[year[0]]: maxs[year[0]] = cells
                         
        
        report(new_map, generateNameT(os.path.join(base, str(i),\
                                        'urb'), year[1]))
        print(year[0], float(maptotal(scalar(new_map))))
        listOfSumStats = metrics.calculateSumStats(new_map, \
                                                        metric_names,\
                                                        zones)

        j=0
        for aname in metric_names:
            observedmap = listOfSumStats[j]
            report(observedmap, \
               generateNameT(os.path.join(base, str(i), aname), \
                            year[1]))
            j+=1
        prev = amap
print('Stochastic area:')        
print(mins)
print(maxs)
print(avs)

