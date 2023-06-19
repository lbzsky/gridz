# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:49:46 2021

@author: lld
"""

#key libraries
import os
import sys
#import pyhdf
import netCDF4
import matplotlib.pyplot as plt #plotting packae
#additional libraries required for definitio]i
import pandas as pd
import numpy as np
from pandas import DataFrame
# import fiona
# import shapefile
# from shapely.geometry import Point
# from itertools import chain 
import geopandas as gpd 
import rasterio as rio
import shapely  

###FUNCTION 1###
# Create a grid based on your input of cell_size and coordinate margins
# lon = longitude in degrees (^o) [int/float with the way how longitude column is called in df]
# lat = latitude in degrees (^o) [int/float, Same as above but latitude]
# cell_size = size in degrees (^o) [int/float]
# def grid_creator(cell_size, min_lon, max_lon, min_lat, max_lat):
#     xmin, ymin, xmax, ymax = min_lon, min_lat, max_lon, max_lat
#     x_interval = np.arange(xmin, xmax, cell_size)
#     y_interval = np.arange(ymin, ymax, cell_size)
#     polygons = []
#     for x in x_interval:
#         for y in y_interval:
#             polygons.append(Polygon([(x,y), (x+cell_size, y), (x+cell_size, y+cell_size), (x, y+cell_size)]))
#     grid = gpd.GeoDataFrame({'geometry':polygons})
#     return grid

def grid_creator(cell_size, min_lon, max_lon, min_lat, max_lat):
    """Creates a grid of polygons from given parameters"""
    xmin, ymin, xmax, ymax = min_lon, min_lat, max_lon, max_lat
    x_interval = np.arange(xmin, xmax, cell_size)
    y_interval = np.arange(ymin, ymax, cell_size)
    polygons = []
    for x_val in x_interval:
        for y_val in y_interval:
            polygons.append(shapely.geometry.Polygon([(x_val, y_val), (x_val + cell_size, y_val),
                                                     (x_val + cell_size, y_val + cell_size),
                                                     (x_val, y_val + cell_size)]))
    grid = gpd.GeoDataFrame({'geometry': polygons})
    return grid

###FUNCTION 2###
# This function creates a grid based on shapefile input you give for convenience
# df = geopandas data frame as input to be gridded
# safe way to open DF is below
# import geopandas as gpd
# df = gpd.read_file(*filename*)
# required format of GF is geopandas.geodataframe.GeoDataFrame
# lon = longitude in degrees (^o) [String with the way how longitude column is called in df]
# lat = latitude in degrees (^o) [String, Same as above but latitude]
# cell_size = size in degrees (^o) [String]

def grid_from_shp(df, lon, lat, cell_size = None):
    gdf = gpd.GeoDataFrame(df, 
                geometry=gpd.points_from_xy(df[lon], df[lat]),
                crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    gdf.drop(df.columns.difference(['orig_ogc_f', 'geometry']), 
             axis=1, inplace=True)#create a geopandas GeoDataFrame object from a pandas DataFrame object, geometry of points = Sinusoidal projection
    xmin, ymin, xmax, ymax= gdf.total_bounds # This line assigns the minimum and maximum coordinates of the bounding box of the GeoDataFrame to xmin, ymin, xmax, and ymax.
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    #this line defines the coordinate reference system as the Sinusoidal projection
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1)  )
    #Loop creates a list of rectangular cells of size cell_size within the bounding box of the GeoDataFrame.
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    return(grid)



def nc(file, grid_df, lon = None, lat = None, varname = None, nanzero = None):
    ds = netCDF4.Dataset(file)

    lon = ds.variables[lon][:]
    lat = ds.variables[lat][:]
    val = ds.variables[varname][:]
    # Generate lon-lat pairs
    # Reshape to desired shape
    output_shape = (len(lat) * len(lon), 1)
    var_reshaped = np.reshape(val, output_shape)
    # Generate lon-lat pairs
    lon_pairs = np.tile(lon, len(lat)).reshape(output_shape)
    lat_pairs = np.repeat(lat, len(lon)).reshape(output_shape)
    # Concatenate all arrays

    rdf = pd.DataFrame({'lon': lon_pairs.squeeze(), 'lat': lat_pairs.squeeze(), varname: var_reshaped.squeeze()})
    rdf = gpd.GeoDataFrame(rdf, geometry=gpd.points_from_xy(rdf.lon, rdf.lat))
    rdf.crs = {'init': 'epsg:4326'}
    
    gdf = gpd.GeoDataFrame(rdf, 
                geometry=gpd.points_from_xy(rdf.lon, rdf.lat))
    gdf.crs = {'init': 'epsg:4326'}
    #gdf = gdf.to_crs(grid_df.crs)
    gdf = gdf.drop(columns=['lon', 'lat'])
    merged_df = gpd.sjoin(grid_df, rdf, how='left', op='intersects')     
    # Group the data by the grid cell index
    if nanzero == 'y':
        index_zero = merged_df[varname]==0
        merged_df[varname][index_zero] = np.nan
    else:
        print('Zeroes will remain in the mean ' +  varname + ' it may affect means')
    grouped_df = merged_df.groupby(merged_df.index)
    # Calculate the mean of the 'var' column for each grid cell
    mean_df = grouped_df[varname].mean().rename(varname + 'av')
    med_df = grouped_df[varname].median().rename(varname + 'md')
    sd_df = grouped_df[varname].std().rename(varname + 'sd')
    # Join the mean, median and std values back to the grid_df dataframe
    grid_output = grid_df.join(mean_df, on=grid_df.index, how='left')
    grid_output = grid_output.join(med_df, on=grid_df.index, how='left')
    grid_output = grid_output.join(sd_df, on=grid_df.index, how='left')
    # Fill any missing values with 0
    grid_output[varname + 'av'].fillna(0, inplace=True)   
    return(grid_output)


### FUNCTION 3, COPYPASTED FUNCTION FROM GPD.SJOIN###
# file - geopandas.series
# grid - grid as geopandas.geodataframe.GeoDataFrame
#'how' specifies the type of spatial join to perform and the available options are 'left', 'right', 'inner' and 'outer'. 
#'op' specifies the spatial relationship between the two geometries and the available options are 'intersects', 'contains', 'within', 'overlaps' and 'touches'.
def shp(file, grid, how = None, op = None):
    merged = gpd.sjoin(grid, file, how="left", op='intersects')
    return(merged)

#FUNCTION 4
#This function is used to rasterize a given file onto a grid
#grouping the data for each cell and returning a dataframe with the mean, median, and standard deviation for each cell. It also allows for the option to replace all 0s with NaN.
#your output will be average, mdeian and sd parameters of a varname 
#varname is just the name of variabile that will be given to your raster as output
#raster_input = tiff file as input 
#grid_df = grid_file(should be geopandas.geodataframe.GeoDataFrame)
def raster(raster_file, grid_df, varname = None, nanzero = None):
    rasty = rio.open(raster_file)
    meta = rasty.meta
    #you calculate x and y future coordinates based on knowledge of width and height of raster
    width = meta['width']
    height = meta['height']
    transform = meta['transform']
    x = np.empty(width * height)
    y = np.empty(width * height)

    #populating x and y arrays using values from raster file     
    for row in range(height):
        for col in range(width):
            x[row * width + col], y[row * width + col] = transform * (col, row)
    val = np.reshape((rasty.read().squeeze()), ((height*width),1))
    
    rdf = pd.DataFrame({'lon': x.flatten(), 'lat': y.flatten(), varname: val.flatten()})
    rdf = gpd.GeoDataFrame(rdf, geometry=gpd.points_from_xy(rdf.lon, rdf.lat))
    rdf.crs = {'init': 'epsg:4326'}

    gdf = gpd.GeoDataFrame(rdf, 
                geometry=gpd.points_from_xy(rdf.lon, rdf.lat))
    gdf.crs = {'init': 'epsg:4326'}
    #gdf = gdf.to_crs(grid_df.crs)
    gdf = gdf.drop(columns=['lon', 'lat'])
    merged_df = gpd.sjoin(grid_df, rdf, how='left', op='intersects')     
    # Group the data by the grid cell index
    if nanzero == 'y':
        index_zero = merged_df[varname]==0
        merged_df[varname][index_zero] = np.nan
    else:
        print('Zeroes will remain in the mean ' +  varname + ' it may affect means')
    grouped_df = merged_df.groupby(merged_df.index)
    # Calculate the mean of the 'var' column for each grid cell
    mean_df = grouped_df[varname].mean().rename(varname + 'av')
    med_df = grouped_df[varname].median().rename(varname + 'md')
    sd_df = grouped_df[varname].std().rename(varname + 'sd')
    # Join the mean, median and std values back to the grid_df dataframe
    grid_output = grid_df.join(mean_df, on=grid_df.index, how='left')
    grid_output = grid_output.join(med_df, on=grid_df.index, how='left')
    grid_output = grid_output.join(sd_df, on=grid_df.index, how='left')
    # Fill any missing values with 0
    grid_output[varname + 'av'].fillna(0, inplace=True)   
    rasty.close()
    return grid_output


 
def date_collocator(strings, ind_month = None):
    import re
    #Defining the regular expression for months and years
    month_pattern = re.compile('(0[1-9]|1[012])')
    year_pattern = re.compile('(199[0-9]|20[0-9][0-9]|21[0-4][0-9])')
    #Iterating over the strings
    mall =[]
    yall =[]
    for string in strings:
        months = month_pattern.findall(string)
        #Adding the condition to this python script so month cannot be located within the same character number of a given string where year was previously identified
        if len(months) > 0 and len(yall) > 0 and len(months) == len(yall):
            if months[-1] != yall[-1][-2:]:
                mall.append(months)
        else:
            mall.append(months)
        years = year_pattern.findall(string)
        yall.append(years) 
    check = (sum([len(a)>1 for a in mall]))
    if check > 0:
        print('ATTENTION YOU HAVE MANY MONTHS, CHOOSE MONTH INDEX AMONG THESE OPTIONS ->')
        print([a-1 for a in list(set([len(a) for a in mall]))])
        try: 
            mall = [b[ind_month] for b in mall]
        except TypeError:
            print('Choose proper index for month, now the output is likely erroneous for months')
    date = [yall[i][0] + '_' + mall[i] for i in range(len(mall)) if yall[i]]
    return {'years':yall,\
            'months':mall,\
            'date':date}
#extract necessary information from metadata

#create empty x and y arrays

#grid_df.to_file(plotpath + '/' + 'gesuga1.shp', driver='ESRI Shapefile')

