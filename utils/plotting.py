import matplotlib.pyplot as plt
import numpy as np

from utils.grid_handlers import coords_to_grid, MI_to_grid
from utils.data_handlers import get_data, closest_coord

# NOTE: Taken from https://github.com/DTU-Nanolab-materials-discovery/intern-data-handling-and-fitting
# All credits to https://github.com/giul-s

# ==== Functions for visualizing XRD data ====

def plot_grid(coords, grid):
    '''"Plot a set of real measurement points on a custom grid defined with the "measurement_grid" function. The corrected grid locations are shown."'''
    corrected_grid = coords_to_grid(coords, grid)
    plt.scatter(grid.iloc[:,0], grid.iloc[:,1], color = 'black', s = 80)
    plt.scatter(coords.iloc[:,0], coords.iloc[:,1], color = 'green', s = 20)
    plt.scatter(corrected_grid.iloc[:,0], corrected_grid.iloc[:,1], color = 'red', s = 20)
    plt.legend(['Defined grid', 'Measured', 'Corrected'])

def plot_data(data, datatype_x, datatype_y, x = "all", y = "all",datatype_select = None,datatype_select_value = None, legend = True, scatter_plot = False,plotscale = "linear", title = "auto"):

    '''Creates a XY plot/scatter plot based on datatype from a dataframe'''

    #x and y to list if only 1 value specified
    if type(x) != list:
        x = [x]
    if type(y) != list:
        y = [y]
    x_data = []
    y_data = []
    labels = []
    #extracts the specified data point by point
    for i in range(len(x)):
        x_data.append(get_data(data, datatype_x, x[i], y[i], False,False))
        y_data.append(get_data(data, datatype_y, x[i], y[i], False,False))
        if x[0] == "all" and y[0] == "all":
            labels = data.columns.get_level_values(0).unique().values
        else:
            grid = MI_to_grid(data)
            xcoord, ycoord = closest_coord(grid, x[i], y[i])
            labels.append('{:.1f},{:.1f}'.format(xcoord, ycoord))
    
    colors = plt.cm.jet(np.linspace(0, 1, len(labels))) #data.columns.get_level_values(0).unique().values

    #formating
    if len(labels) == 1:
        labels = labels[0]
    if x[0] == "all" and y[0] == "all":
        x_data = x_data[0]
        y_data = y_data[0]
    else:
        x_data = np.transpose(x_data)
        y_data = np.transpose(y_data)
    
    #if datatype with multiple values per point is plotted only selects one value, based on the datatype_select, datatype_select_value. 
    if datatype_select != None:
        y_data = y_data.iloc[data.index[data[data.iloc[:, data.columns.get_level_values(1)== datatype_select].columns[0]] == datatype_select_value]]
        x_data_coords = x_data.columns.get_level_values(0)
        y_data_coords = y_data.columns.get_level_values(0)
        data_coords = [j for j in x_data_coords if j not in y_data_coords]
        x_data.drop(data_coords, level=0,axis = 1, inplace=True) 
        x_data = x_data.values[0]
        y_data = y_data.values[0]
        labels = datatype_select + ': ' + str(round(datatype_select_value,2))

#plots scatter plot if scatter_plot is not false, else line plot
    if x[0] == "all" and y[0] == "all":
        for idx, (x_val, y_val) in enumerate(zip(x_data.values.T, y_data.values.T)):
            if scatter_plot:
                plt.plot(x_val, y_val, 'o', color=colors[idx], label=labels[idx])
            else:
                plt.plot(x_val, y_val, color=colors[idx], label=labels[idx])
    else:
        for idx, (x_val, y_val) in enumerate(zip(x_data.T, y_data.T)):
            if scatter_plot:
                plt.plot(x_val, y_val, 'o', color=colors[idx], label=labels[idx])
            else:
                plt.plot(x_val, y_val, color=colors[idx], label=labels[idx])
    plt.xlabel(datatype_x)
    plt.ylabel(datatype_y)
    plt.yscale(plotscale)
    if legend == True:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title == "auto":
        plt.title("{} over {}".format(datatype_y, datatype_x))
    else:
        plt.title(title)