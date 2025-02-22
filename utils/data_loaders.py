import numpy as np
import pandas as pd
import re
from utils.grid_handlers import (
    coords_to_grid,
    grid_to_MIheader,
)


# NOTE: Taken from https://github.com/DTU-Nanolab-materials-discovery/intern-data-handling-and-fitting
# All credits to https://github.com/giul-s

# ==== Functions for handling Data ====

def read_layerprobe(filename, grid, sheetname = -1, n = 0):
    '''"Read data and coordinates from a LayerProbe datafile. The file should be an Excel sheet. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_layerprobe(filename) Optional: "sheetname" name of sheet, defaults to last sheet in file. "n" - amount of measurements to include.'''
   # read data and limit length based on amount of wanted points
    data = pd.read_excel(filename, sheet_name = sheetname, header=0)
    if n > 0:
        data = data.truncate(after=n-1)
    data = data.sort_values(by=['X (mm)','Y (mm)'])
    data = data.reset_index(drop = True)
    # we need coords for aligning data to grid
    # extract coordinates
    coords = data.copy()
    coords = coords.iloc[:,1:3]
    coords.rename(columns={"X (mm)": "x", "Y (mm)": "y"}, inplace=True)
    # treat coordinates
    coords = coords.astype(float)
    coords = coords.round(4)

    # remove coordinates from data
    data = data.drop(data.columns[0:3], axis = 1)

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, data],names=['Coordinate','Data type'])
    data = pd.DataFrame(data.to_numpy().flatten()).transpose()
    data = pd.DataFrame(data.values, columns=header)
    return data, coords

def read_XRD(filename, grid, n = 0, separator = "\t"):
    '''"Read data from an XRD datafile into a dataframe. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_XRD(filename) Optional: "n" - amount of measurements to include. "separator" - csv file separator.'''
    # read data and limit length based on amount of wanted points
    data = pd.read_csv(filename, sep=separator, header=1)
    if n > 0:
        data = data.iloc[:,0:n*2]
    data.rename(columns={"2θ, °": "2θ (°)", "Intensity, counts": "Intensity (counts)"}, inplace=True)

    # we need coords for aligning data to grid
    # only load row of measurement names, and convert to an array of those names
    file_header = pd.read_csv(filename, sep=separator, header=0, nrows=0)
    coords_array = file_header.columns.values[::2]
    # limit length based on amount of wanted points
    if n > 0:
        coords_array = coords_array[0:n]
    
    # extract coordinate info from headers
    for i in range(len(coords_array)):
        # split header and select coordinates
        split_list = re.split('_', coords_array[i])
        coords_array[i] = split_list[-2:]

        # replace '-' character with '.', but preserve '-' at start for negative numbers
        for j in range(2):
            coords_array[i][j] = re.sub('(?!^)-', '.', coords_array[i][j])

    # convert array to a list otherwise Pandas does not work
    coords_list = list(coords_array)
    coords = pd.DataFrame(coords_list, columns=['x', 'y'])

    # do some treatment on the dataframe
    coords = coords.astype(float)

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct XRD dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, data.columns[0:2]],names=['Coordinate','Data type'])
    data = pd.DataFrame(data.values, columns=header)
    return data, coords

def read_ellipsometry_thickness(filename, grid, n = 0, separator = "\t"):
    '''"Read thickness data and coordinates from an ellipsometry datafile. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_ellipsometry_thickness(filename) Optional: "n" - amount of measurements to include. "separator" - csv file separator.'''
    # read data and limit length based on amount of wanted points
    data = pd.read_csv(filename, sep=separator, header=1)
    if n > 0:
        data = data.truncate(after=n-1)
    data.rename(columns={"Z": "Z (nm)"}, inplace=True)

    # we need coords for aligning data to grid
    # extract coordinates
    coords = data.copy()
    coords = coords.drop(columns=['Z (nm)'])
    coords.rename(columns={"X (cm)": "x", "Y (cm)": "y"}, inplace=True)
    # convert to float
    coords = coords.astype(float)
    # convert from cm to mm
    coords = coords*10

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct ellipsometry dataframe with multiindexing for coordinates
    data = data.drop(columns=['X (cm)','Y (cm)'])
    data = data.stack().to_frame().T
    # "verify_integrity = False" lmao
    data.columns = data.columns.set_levels(coord_header, level=0, verify_integrity=False)
    data.columns.rename(["Coordinate", "Data type"], level=[0, 1], inplace = True)
    return data, coords

def read_ellipsometry_MSE(filename, grid, n = 0, separator = "\t"):
    '''"Read Mean Squared Error data and coordinates from an ellipsometry datafile. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_ellipsometry_MSE(filename) Optional: "n" - amount of measurements to include. "separator" - csv file separator.'''
    # read data and limit length based on amount of wanted points
    data = pd.read_csv(filename, sep=separator, header=1)
    if n > 0:
        data = data.truncate(after=n-1)
    data.rename(columns={"Z": "MSE"}, inplace=True)

    # we need coords for aligning data to grid
    # extract coordinates
    coords = data.copy()
    coords = coords.drop(columns=['MSE'])
    coords.rename(columns={"X (cm)": "x", "Y (cm)": "y"}, inplace=True)
    # convert to float
    coords = coords.astype(float)
    # convert from cm to mm
    coords = coords*10

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct ellipsometry dataframe with multiindexing for coordinates
    data = data.drop(columns=['X (cm)','Y (cm)'])
    data = data.stack().to_frame().T
    # "verify_integrity = False" lmao
    data.columns = data.columns.set_levels(coord_header, level=0, verify_integrity=False)
    data.columns.rename(["Coordinate", "Data type"], level=[0, 1], inplace = True)
    return data, coords

def read_ellipsometry_nk(filename, grid, n = 0, separator = "\t"):
    '''"Read refractive index n and absorption coefficient k data and coordinates from an ellipsometry datafile. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_ellipsometry_nk(filename) Optional: "n" - amount of measurements to include. "separator" - csv file separator.'''
    # read data and split into energy and n/k data, limited by number of wanted points
    data = pd.read_csv(filename, sep=separator, header=1, index_col = False)
    data_energy = data.iloc[:,0]
    if n > 0:
        data_n_k = data.iloc[:,1:n*2+1]
    else:
        data_n_k = data.iloc[:,1:]

    # get headers from n/k data to get an array of coordinates
    coords_array = np.array(data_n_k.columns)

    # extract coordinate info from headers
    for i in range(len(coords_array)):
        # split header and select coordinates
        split_list = re.split(',', coords_array[i])
        split_list[0] = float(split_list[0][4:])
        split_list[1] = float(split_list[1][:-1])
        coords_array[i] = split_list
    coords_array = coords_array[::2]

    # convert array to a list otherwise Pandas does not work
    coords_list = list(coords_array)
    coords = pd.DataFrame(coords_list, columns=['x', 'y'])

    # convert from cm to mm
    coords = coords*10

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # rename all n and k columns, and insert energy column before each n and k set
    k = 0
    for i in range(0, len(data_n_k.columns), 2):
        data_n_k.columns.values[i+k] = "n"
        data_n_k.columns.values[i+k+1] = "k"
        data_n_k.insert(loc=i+k, column='Energy (eV)'.format(i), value=data_energy, allow_duplicates = True)
        k += 1

    # construct XRD dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, data.columns[0:3]],names=['Coordinate','Data type'])
    data = pd.DataFrame(data_n_k.values, columns=header)
    return data, coords

def convert_to_eV(data):
    '''"Convert ellipsometry data in wavelength to eV."'''
    # Planck's constant (eV/Hz)
    h = 4.135*10**-15
    # Speed of light (m/s)
    c = 3*10**8
    data = data.copy()
    data.iloc[:, data.columns.get_level_values(1) == 'Wavelength (nm)'] = (h*c)/(data.iloc[:, data.columns.get_level_values(1) == 'Wavelength (nm)']*10**-9)
    data.columns = data.columns.set_levels(['Energy (eV)', 'k', 'n'], level=1)
    data = data.round(3)
    return data

def read_XPS(filename, grid):
    '''"Read data and coordinates from an XPS datafile. The file should be an csv (.txt) file. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_XPS(filename, grid)"'''
    # read the file
    file = pd.read_csv(filename, encoding = 'ANSI', engine='python', sep='delimiter', header = None, skiprows = 29)
    file.drop(file.iloc[4::7].index, inplace=True)
    file.reset_index(drop = True)

    # the file has a really weird format so we need to do a lot of work to extract data
    # get amount of peaks
    peaknumb = []
    for i in range(0, len(file), 6):
        peaknumb.append(int(file.iloc[i][0].split()[8].replace(";","")))
    n = max(peaknumb) + 1

    # remove useless rows
    file.drop(file.iloc[0::6].index, inplace=True)
    file.reset_index(drop = True)

    # get data from remaining rows
    full_peaklist = []
    peaklist = []
    coordlist = []
    datalist = []
    for i in range(0, len(file), 5):
        # load peak type and coordinates and fix formatting
        peaktype = ' '.join(file.iloc[i][0].split()[5:len(file.iloc[i][0].split())]).replace("VALUE='","").replace("';","")
        xcoord = float(file.iloc[i+1][0].split()[5].replace("VALUE=","").replace(";",""))
        ycoord = float(file.iloc[i+2][0].split()[5].replace("VALUE=","").replace(";",""))
        coords = [xcoord, ycoord]
        # load data
        data = file.iloc[i+3][0].split()[2::]
        data.append(file.iloc[i+4][0].split()[2::][0])
        # fix data formatting
        data = [j.replace(",","") for j in data]
        data = [round(float(j),3) for j in data]

        full_peaklist.append(peaktype)
        peaklist.append(peaktype.split()[0])
        coordlist.append(coords)
        datalist.append(data)

    # create data dataframe
    dataframe = pd.DataFrame(datalist, columns = ['Intensity (counts)','Atomic %','Area (counts*eV)','FWHM (eV)','Peak BE (eV)'])
    # modify some values
    # convert cps to counts (machine does 25 cps)
    dataframe['Intensity (counts)'] = dataframe['Intensity (counts)']/25
    # convert KE to BE (KE of machine X-rays is 1486.68 eV)
    dataframe['Peak BE (eV)'] = 1486.68 - dataframe['Peak BE (eV)']
    # reorder columns to be similar to Avantage
    columnorder = ['Peak BE (eV)','Intensity (counts)','FWHM (eV)','Area (counts*eV)','Atomic %']
    dataframe = dataframe.reindex(columnorder, axis=1)

    # create coordinate dataframe
    coords = pd.DataFrame(coordlist, columns=['x', 'y'])
    # remove duplicate coordinates
    coords = coords.drop_duplicates(ignore_index = True)
    # adjust range to center coords on 0,0 instead of upper left corner
    coords['x'] = coords['x'] - max(coords['x'])/2
    coords['y'] = coords['y'] - max(coords['y'])/2
    # convert coords from µm to mm
    coords = coords/1000
    # flip y coordinate because Avantage is mental
    coords['y'] = coords['y'].values[::-1]

    # create peak dataframe
    peaks = pd.DataFrame(peaklist, columns = ['Peak'])
    # add peak dataframe to front of data dataframe
    dataframe = pd.concat([peaks, dataframe], axis = 1)

    # add column with summed atomic %
    element_list = dataframe['Peak'].unique()
    atomic_percent_list = []
    for l in range(0, int(len(peaklist)), n):
        for k in range(len(element_list)):
            atomic_percent = round(sum(dataframe.iloc[l:l+n].loc[dataframe['Peak'] == element_list[k]]["Atomic %"]),3)
            atomic_percent_list.append(atomic_percent)
        for j in range(len(element_list)*(n-1)):
            atomic_percent_list.append(float("NaN"))
    atomic_percent_array = np.split(np.array(atomic_percent_list), len(atomic_percent_list)/len(element_list))
    atomic_percent_frame = pd.DataFrame(atomic_percent_array, columns=element_list + " Total")
    dataframe = pd.concat([dataframe, atomic_percent_frame], axis=1)

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct XRD dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, dataframe.columns],names=['Coordinate','Data type'])
    # reorder dataframe stacking to fit coordinate attachment
    n2 = n
    stackedframe = np.hstack([dataframe.values[0:n2],(dataframe.values[n2:2*n2])])
    for i in range(2*n2, len(dataframe), n2):
        stackedframe = np.hstack([stackedframe, (dataframe.values[i:i+n2])])
    data = pd.DataFrame(stackedframe, columns=header)
    return data, coords


def read_UPS(filename, grid):
    '''"Read data and coordinates from an UPS datafile. The file should be an Excel (.xlsx) file. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_UPS(filename, grid)"'''
    # load data, energy, and coordinates from all sheets
    dataload_counts = pd.read_excel(filename, sheet_name = None, skiprows = 18)
    dataload_eV = pd.read_excel(filename, sheet_name = None, skiprows = 18, usecols = [0])
    coordload = pd.read_excel(filename, sheet_name = None, skiprows = 13, nrows = 3)

    # select dictionary keys for only usable sheets
    dictlist = list(dataload_eV.keys())

    # remove useless "Titles" sheets that Avantage generates
    # also remove "Peak Table" sheets as we get coordinates from graph sheets
    j = 0
    for i in range(len(dictlist)):
        if dictlist[j].startswith("Titles"):
            del dictlist[j]
            j -= 1
        if dictlist[j].startswith("Peak Table"):
            del dictlist[j]
            j -= 1
        j += 1

    # remove last sheet in file as it should be a blank sheet
    del dictlist[-1]

    data_list = []
    xy_coords_list = []

    # read data
    for i in range(0, len(dictlist)):
        # load data from usable sheets only
        dataselect_counts = dataload_counts[dictlist[i]].dropna(axis = 1, how = 'all')
        data = dataselect_counts.iloc[:,1:]
        data_eV = dataload_eV[dictlist[i]]
        # rename columns
        data.columns.values[:] = "Intensity (counts)"
        # insert energy column besides each data column
        j = 0
        for k in range(0, len(data.columns)):
            data.insert(loc=k+j, column='BE (eV)'.format(k), value=data_eV, allow_duplicates = True)
            j += 1
        data_list.append(data)

        # read coords
        coordselect = coordload[dictlist[i]].dropna(axis=1, how = 'all')
        xcoord = coordselect.iloc[0,1]
        ycoord = coordselect.iloc[2,2:]
        # create coords list
        for l in range(len(ycoord)):
            x = xcoord
            y = ycoord[l]
            xy_coords = np.array([x, y])
            xy_coords_list.append(xy_coords)

    # create coords dataframe
    coords = pd.DataFrame(xy_coords_list, columns=['x', 'y'])

    # create merged data dataframe
    dataframe = pd.concat(data_list, axis = 1)

    # adjust range to center coords on 0,0 instead of upper left corner
    coords['x'] = coords['x'].astype(float)# - max(coords['x'].astype(float))/2
    coords['y'] = coords['y'].astype(float)# - max(coords['y'].astype(float))/2

    # convert coords from µm to mm
    coords = coords/1000

    # return dataframe, coords

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct XRD dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, data.columns.unique()],names=['Coordinate','Data type'])
    data = pd.DataFrame(dataframe.values, columns=header)
    return data

def read_UPS_old(filename, grid):
    '''"Read data and coordinates from an UPS datafile. The file should be an Excel (.xlsx) file. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_UPS(filename, grid)"'''
    # load data, energy, and coordinates from all sheets
    dataload_counts = pd.read_excel(filename, sheet_name = None, skiprows = 18)
    dataload_eV = pd.read_excel(filename, sheet_name = None, skiprows = 18, usecols = [0])
    coordload = pd.read_excel(filename, sheet_name = None, skiprows = 13, nrows = 3)

    # select dictionary keys for only usable sheets
    dictlist = list(dataload_eV.keys())

    # remove useless "Titles" sheets that Avantage generates
    # also remove "Peak Table" sheets as we get coordinates from graph sheets
    j = 0
    for i in range(len(dictlist)):
        if dictlist[j].startswith("Titles"):
            del dictlist[j]
            j -= 1
        if dictlist[j].startswith("Peak Table"):
            del dictlist[j]
            j -= 1
        j += 1

    # remove last sheet in file as it should be a blank sheet
    del dictlist[-1]

    data_list = []
    xy_coords_list = []

    # read data
    for i in range(0, len(dictlist)):
        # load data from usable sheets only
        dataselect_counts = dataload_counts[dictlist[i]].dropna(axis = 1, how = 'all')
        data = dataselect_counts.iloc[:,1:]
        data_eV = dataload_eV[dictlist[i]]
        # rename columns
        data.columns.values[:] = "Intensity (counts)"
        # insert energy column besides each data column
        j = 0
        for k in range(0, len(data.columns)):
            data.insert(loc=k+j, column='BE (eV)'.format(k), value=data_eV, allow_duplicates = True)
            j += 1
        data_list.append(data)

        # read coords
        coordselect = coordload[dictlist[i]].dropna(axis=1, how = 'all')
        xcoord = coordselect.iloc[0,1]
        ycoord = coordselect.iloc[2,2:]
        # create coords list
        for l in range(len(ycoord)):
            x = xcoord
            y = ycoord[l]
            xy_coords = np.array([x, y])
            xy_coords_list.append(xy_coords)

    # create coords dataframe
    coords = pd.DataFrame(xy_coords_list, columns=['x', 'y'])

    # create merged data dataframe
    dataframe = pd.concat(data_list, axis = 1)

    # adjust range to center coords on 0,0 instead of upper left corner
    # coords['x'] = coords['x'] - max(coords['x'])/2
    # coords['y'] = coords['y'] - max(coords['y'])/2

    # convert coords from µm to mm
    coords = coords/1000

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct XRD dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, data.columns.unique()],names=['Coordinate','Data type'])
    data = pd.DataFrame(dataframe.values, columns=header)
    return data, coords

def read_REELS(filename, grid):
    '''"Read data and coordinates from an REELS datafile. The file should be an Excel (.xlsx) file. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_REELS(filename, grid)"'''
    # This data loading method is identical to UPS.
    data, coords = read_UPS(filename, grid)
    return data, coords

def read_raman(filename, grid):
    '''"Read data and coordinates from a Raman spectroscopy datafile. The file should be an Excel (.xlsx) file. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_raman(filename)"'''
    # load data, energy, and coordinates from all sheets
    dataload_counts = pd.read_excel(filename, sheet_name = None, skiprows = 18)
    dataload_eV = pd.read_excel(filename, sheet_name = None, skiprows = 18, usecols = [0])
    coordload = pd.read_excel(filename, sheet_name = None, skiprows = 13, nrows = 3)

    # select dictionary keys for only usable sheets
    dictlist = list(dataload_eV.keys())

    # remove useless "Titles" sheets that Avantage generates
    # also remove "Peak Table" sheets as we get coordinates from graph sheets
    j = 0
    for i in range(len(dictlist)):
        if dictlist[j].startswith("Titles"):
            del dictlist[j]
            j -= 1
        if dictlist[j].startswith("Peak Table"):
            del dictlist[j]
            j -= 1
        j += 1

    # remove last sheet in file as it should be a blank sheet
    del dictlist[-1]

    data_list = []
    xy_coords_list = []

    # read data
    for i in range(0, len(dictlist)):
        # load data from usable sheets only
        data = dataload_counts[dictlist[i]].iloc[:,2:]
        data_eV = dataload_eV[dictlist[i]]
        # rename columns
        data.columns.values[:] = "Intensity (counts)"
        # insert energy column besides each data column
        j = 0
        for k in range(0, len(data.columns)):
            data.insert(loc=k+j, column='Raman shift (cm^-1)'.format(k), value=data_eV, allow_duplicates = True)
            j += 1
        data_list.append(data)

        # read coords
        xcoord = coordload[dictlist[i]].iloc[0,1]
        ycoord = coordload[dictlist[i]].iloc[2,2:]
        # create coords list
        for l in range(len(ycoord)):
            x = xcoord
            y = ycoord[l]
            xy_coords = np.array([x, y])
            xy_coords_list.append(xy_coords)

    # create coords dataframe
    coords = pd.DataFrame(xy_coords_list, columns=['x', 'y'])

    # create merged data dataframe
    dataframe = pd.concat(data_list, axis = 1)

    # adjust range to center coords on 0,0 instead of upper left corner
    coords['x'] = coords['x'] - max(coords['x'])/2
    coords['y'] = coords['y'] - max(coords['y'])/2

    # convert coords from µm to mm
    coords = coords/1000

    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct XRD dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, data.columns.unique()],names=['Coordinate','Data type'])
    data = pd.DataFrame(dataframe.values, columns=header)
    return data, coords