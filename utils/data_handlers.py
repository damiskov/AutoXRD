import pandas as pd
import numpy as np
from utils.grid_handlers import MI_to_grid, grid_to_MIheader
from IPython.display import display



# NOTE: Taken from https://github.com/DTU-Nanolab-materials-discovery/intern-data-handling-and-fitting
# All credits to https://github.com/giul-s

# === Functions for Materials Science Data ===


def add_info(data, info_dict):
    """Function to add information to a dataset for each point."""
    info_type = list(info_dict.keys())[0]
    headerlength = len(data.columns.get_level_values(1).unique())
    coords= data.columns.get_level_values(0).unique()
    k=0
    new_data = data.copy()
    for i in range(0, len(coords)):
        #print(coords[i])
        new_df = pd.DataFrame([info_dict[info_type][i]], columns =[(coords[i], info_type)])
            
        new_data.insert(headerlength*(i+1)+k, "{}".format(data.columns.get_level_values(0).unique()[i]), new_df, allow_duplicates=True)
        new_data.rename(columns={'':  f'{info_type}'}, inplace = True)

        k=k+len(new_df.columns)
                
    new_frame = new_data.copy()

    return new_frame

def closest_coord(grid, x, y):
    '''"Find closest x and y coordinate for a grid."'''
    xminindex = np.abs(grid - x).idxmin().iloc[0]
    xcoord = grid.iloc[xminindex, 0]
    yminindex = np.abs(grid[grid['x']==xcoord] - y).idxmin().iloc[1]
    ycoord = grid.iloc[yminindex, 1]
    return xcoord, ycoord

def combine_data(datalist):
    '''"Combine multiple measurements into a single dataframe."'''
    dataframe = pd.concat(datalist, axis=1)
    return dataframe

def math_on_columns(data, type1, type2, operation = "/"):
    '''"Perform an operation on two columns in a provided dataframe. Usage: math_on_columns(data, datatype1, datatype2, operation), where "operation" can be +, -, *, or /."'''
    coordinatelength = len(data.columns.get_level_values(0).unique())
    headerlength = len(data.columns.get_level_values(1).unique())
    k = 0
    # do math on values
    data = data.copy()
    for i in range(coordinatelength):
        val1 = data.iloc[:, data.columns.get_level_values(1)==type1].iloc[:,i]
        if isinstance(type2, str):
            val2 = data.iloc[:, data.columns.get_level_values(1)==type2].iloc[:,i]
        if isinstance(type2, (int,float)):
            val2= type2
        if operation == "+":
            resultval = val1 + val2
        elif operation == "-":
            resultval = val1 - val2
        elif operation == "*":
            resultval = val1 * val2
        elif operation == "/":
            try:
                resultval= val1 / val2
            except ZeroDivisionError:
                resultval = float("NaN")
        # insert result
        data.insert(headerlength*(i+1)+k, "{}".format(data.columns.get_level_values(0).unique()[i]), resultval, allow_duplicates=True)
        k += 1

    # rename added columns
    if operation == "+":
        rowname = "{} + {}".format(type1, type2)
    elif operation == "-":
        rowname = "{} - {}".format(type1, type2)
    elif operation == "*":
        rowname = "{} * {}".format(type1, type2)
    elif operation == "/":
        rowname = "{} / {}".format(type1, type2)
    data.rename(columns={'':rowname}, inplace = True)
    return data

def get_data(data, type = 'all', x = 'all', y = 'all', printinfo = True, drop_nan = True):
    '''"Print a data type from a multi index dataframe at a specific coordinate. The coordinate does not have to be exact. Leave type as blank or 'all' to select all types. Leave coordinates blank or 'all' to select all coordinates."'''
    if x == 'all' and y == 'all':
        if type == 'all':
            if printinfo == True:
                print("All data at all coordinates.")
            if drop_nan == True:
                data = data.dropna(axis = 0, how = 'all').fillna('-')
            return data
        else:
            if printinfo == True:
                print("{} data at all coordinates.".format(type))
            if drop_nan == True:
                data = data.dropna(axis = 0, how = 'all').fillna('-')
            return data.iloc[:, data.columns.get_level_values(1)==type]
    else:
        datagrid = MI_to_grid(data)
        # find closest x and y coordinate
        xcoord, ycoord = closest_coord(datagrid, x, y)
        coords = '{},{}'.format(xcoord, ycoord)
        if type == 'all':
            if printinfo == True:
                print("All data at {},{}.".format(x, y))
            if drop_nan == True:
                data = data.dropna(axis = 0, how = 'all').fillna('-')
            return data.xs(coords, axis=1)
        else:
            if printinfo == True:
                print("{} data at {},{}.".format(type, x, y))
            if drop_nan == True:
                data = data.dropna(axis = 0, how = 'all').fillna('-')
            return data.xs(coords, axis=1)[type]
        
def translate_data(data, x, y):
    '''"Move a set of datapoints by a given x and y offset. Useful when combining multiple samples into one dataframe."'''
    coords = MI_to_grid(data)
    coords['x'] = coords['x'] + x
    coords['y'] = coords['y'] + y
    coord_header = grid_to_MIheader(coords)
    header = pd.MultiIndex.from_arrays([coord_header, data.columns.get_level_values(1)],names=['Coordinate','Data type'])
    data = pd.DataFrame(data.values, columns=header)
    return data, coords

def save_data(dataframe, filename, separator = "\t"):
    '''"Save dataframe to tab seperated txt file."'''
    dataframe.to_csv(filename, separator, index=False, encoding='utf-8')
    return

def load_data(filepath, separator = "\t"):
    '''"Load txt to dataframe."'''
    dataframe = pd.read_csv(filepath, sep=separator, header=[0, 1])
    dataframe.columns.rename(["Coordinate", "Data type"], level=[0, 1], inplace = True)
    return dataframe

def export_specific(data, type, x, y, path): 
    'export a specific point in XY format in a .txt file'
    data_exp = get_data(data, type=type, x=x, y=y)
    data_exp.to_csv(path, sep='\t', index=False, header=False)
    display(data_exp) 