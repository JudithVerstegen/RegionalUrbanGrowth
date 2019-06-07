from osgeo import ogr
import os

### Script for selecting train stations from OSM transport data. Data was downlowaded for Ireland, Italy and Poland.
### Data is in folders with names corresponding to the names of the countries
### railway=halt and railway=station

def create_filtered_shapefile(in_shapefile, country, data_dir):
    print('Reading in the shapefile...')
    # Get the correct driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # 0 means read-only. 1 means writeable.
    data_source = driver.Open(in_shapefile,0)
    print('Shapefile read...')

    # Check to see if shapefile is found.
    if data_source is None:
        print('Could not open %s' % (in_path))
    
    # get the Layer class object
    input_layer = data_source.GetLayer(0)

    # Filter by our query
    print("Filtering by query...")
    query_str = "fclass = 'railway_station' OR fclass = 'railway_halt'"
    print('Query created...')

    # Apply a filter
    input_layer.SetAttributeFilter(query_str)
    print('input layer filtered')
    
    # Set output dir and name
    out_shapefile = data_dir+'/railways/stations_'+country+'.shp'
    print('Name of the output assigned...')

    # Copy Filtered Layer and Output File
    driver = ogr.GetDriverByName('ESRI Shapefile')
    print('Driver read...')
    out_ds = driver.CreateDataSource(out_shapefile)
    print('Out data source created...')
    out_layer = out_ds.CopyLayer(input_layer, 'station')
    
    del input_layer, out_layer, out_ds
    return out_shapefile

# Data directory
data_dir = os.path.join('D:\\', 'Nauka', 'Geobazy', \
'CORINE', 'Student_Assistant_Judith', 'from_Judith', 'RegionalUrbanGrowth', 'RegionalUrbanGrowth','data')

# Select the shapefile
files = os.listdir(os.path.join(data_dir, 'railways'))
print(files)

for f in files:
    print('Current country: ', f)
    in_fn = os.path.join(data_dir, 'railways', f, 'gis_osm_transport_free_1.shp')
    print(in_fn)
    filtered = create_filtered_shapefile(in_fn, f, data_dir)
    print('filtered shapefile created')
    
