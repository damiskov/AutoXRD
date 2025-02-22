import numpy as np
import pandas as pd
import re
from scipy.interpolate import griddata


# NOTE: Taken from https://github.com/DTU-Nanolab-materials-discovery/intern-data-handling-and-fitting
# All credits to https://github.com/giul-s


# === Functions for handling grids ===

def measurement_grid(ncolumns, nrows, gridlength, gridheight, startcolumn = 0, startrow = 0):
    '''"Define a grid based on number of columns and rows, length and height of grid in mm, and the first coordinate (lower left corner) in the column and row."'''
    xgrid = np.round(np.linspace(startcolumn, gridlength+startcolumn, ncolumns), 3)
    ygrid = np.round(np.linspace(startrow, gridheight+startrow, nrows), 3)
    grid = np.array([xgrid[0], ygrid[0]])
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            grid = np.vstack((grid, np.array([xgrid[i], ygrid[j]])))
    grid = grid[1:]
    grid = pd.DataFrame(grid, columns = ['x','y'])
    return grid

def coords_to_grid(coords, grid):
    '''"Constrain a set of measured datapoints to a custom defined grid made with the "measurement_grid" function."'''
    griddata = coords.copy()
    for i in range(len(coords)):
        # find closest x and y coordinate
        xminindex = np.abs(grid - coords.iloc[i,:]).idxmin().iloc[0]
        yminindex = np.abs(grid - coords.iloc[i,:]).idxmin().iloc[1]
        # assign new coordinates
        griddata.iloc[i,0] = np.round(grid.iloc[xminindex, 0], 2)
        griddata.iloc[i,1] = np.round(grid.iloc[yminindex, 1], 2)
    return(griddata)

def grid_to_MIheader(grid):
    '''"Convert a grid (array of x,y) into a multi index header"'''
    MIgrid = []
    for i in range(len(grid)):
        MIgrid = np.append(MIgrid, ('{},{}'.format(grid.iloc[i,0], grid.iloc[i,1])))
    return MIgrid

def MI_to_grid(MIgrid):
    '''"Convert multi index into a grid (array of x,y)"'''
    MIgrid = MIgrid.columns.get_level_values(0)
    splitvals = re.split(',', MIgrid[0])
    grid = np.array([splitvals[0], splitvals[1]])
    for i in range(1, len(MIgrid)):
        splitvals = re.split(',', MIgrid[i])
        grid = np.vstack((grid, np.array([splitvals[0], splitvals[1]])))
    grid = pd.DataFrame(grid, columns = ['x','y'])
    grid = grid.astype(float)
    return grid

def interpolate_grid(data, grid):
    '''"Interpolate data over a custom grid made with the "measurement_grid" function."'''
    # !!!
    # specifically remove the "Peak" column, which will be present if loaded data is XPS
    # !!!
    data = data.drop(columns = 'Peak', level=1, errors = 'ignore')

    # get grid-aligned coordinates for datapoints for interpolation
    coords = MI_to_grid(data).drop_duplicates(ignore_index=True)

    # interpolation
    # we have to account for multiple variables for every coordinate
    # this creates a list of the data
    dataT = data.transpose()
    dataN = int(len(dataT)/len(coords))
    interpolated_data = []
    for i in range(dataN):
        interpolated_data.append(griddata(coords, dataT.iloc[i::dataN], grid, method = "cubic"))

    # convert the list of data to an array with columns alternating between data types
    if dataN == 0:
        interpolated_array = np.transpose(interpolated_data)
    else:
        interpolated_array = [None]*len(interpolated_data[0])*dataN
        for i in range(dataN):
            interpolated_array[i::dataN] = interpolated_data[i]
        interpolated_array = np.transpose(interpolated_array)

    # list of column names
    columnlist = []
    for i in range(interpolated_array.shape[1]):
        columnlist.append(data.columns[i%dataN][1])

    # dataframe
    interpolated_frame = pd.DataFrame(interpolated_array, columns = columnlist)

    # convert grid to multiindex header
    coord_header = grid_to_MIheader(grid)

    # construct dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, list(interpolated_frame.columns.unique())], names=['Coordinate', 'Data type'])
    interpolated_frame = pd.DataFrame(interpolated_frame.values, columns=header)
    return interpolated_frame

def extract_coordinates(data):
    coords= data.columns.get_level_values(0).unique().values
    x_values = []
    y_values = []
    
    for item in coords:
        x, y = item.split(',')
        x_values.append(float(x))
        y_values.append(float(y))
    
    return x_values, y_values

def snake_grid(x, y): # x and y are lists of coordinates you should take note of 
    """ Create a snake grid from x and y coordinates, for CRAIC data. """

    X_snake = []
    Y_snake = []

    # Loop through each y-coordinate from bottom to top
    for i, y_val in enumerate(y):

        if i % 2 == 0: 
            X_snake.extend(x) # Even row: left to right ( add x normally)
        else: 
            X_snake.extend(x[::-1]) # Odd row: right to left (add x in reverse)
        Y_snake.extend([y_val] * len(x)) # add as many y values as x values

    grid_snake = pd.DataFrame({"x": X_snake, "y": Y_snake})
    return grid_snake

def select_points(data, x_min=-40, x_max=40, y_min=-40, y_max=40):
    'get coordinates of the points within the defined range, you can call them with get_data, or plot_data, or interactive_XRD_shift'
    grid0 = MI_to_grid(data)
    grid = grid0.drop_duplicates().reset_index(drop=True)

    grid1 = grid[grid['x'] >x_min]
    grid2 = grid1[grid1['x'] <x_max]
    grid3 = grid2[grid2['y'] >y_min]
    grid4 = grid3[grid3['y'] <y_max]
    new_x = grid4['x'].values
    new_y = grid4['y'].values
    return new_x, new_y

def rotate_coordinates(data_df, how ='clockwise'):
    'Rotate the coordinates of the data by 90 degrees clockwise, counterclockwise or 180 degrees'
    MI_rotated=[]
    initial_coordinates = MI_to_grid(data_df)

    if how == 'clockwise':
        xx = initial_coordinates['y']
        yy = - initial_coordinates['x']

    if how == 'counterclockwise':
        xx = - initial_coordinates['y']
        yy = initial_coordinates['x']

    if how == '180':
        xx = - initial_coordinates['x']
        yy = - initial_coordinates['y']

    for i in range(len(xx)):
        MI_rotated = np.append(MI_rotated,('{},{}'.format(xx[i], yy[i])))
    rotated_columns = pd.MultiIndex.from_tuples([(str(coord), col) for coord, col in zip(MI_rotated, data_df.columns.get_level_values(1))])
    data_rotated = data_df.copy()
    data_rotated.columns = rotated_columns
    return data_rotated