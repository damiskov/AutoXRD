
# NOTE: Taken from https://github.com/DTU-Nanolab-materials-discovery/intern-data-handling-and-fitting
# All credits to https://github.com/giul-s


# === Functions for EDX Analysis ===

def EDX_stage_coords(folder, filename):
    'Calculate EDX coordinates for a given file. Requires .xlsx file with columns as in template.'

    # Define file paths
    filepath = os.path.join(folder, filename + ".xlsx")
    newpath = os.path.join(folder, filename + "_stage_coords.xlsx")
    
    # Read specific columns from the Excel file, and extract values
    file = pd.read_excel(filepath, sheet_name='Sheet2', usecols='H:P')
    file = file.drop(file.index[3:])  # Drop the 4th row (assumes you have info for 3 points)

    nrows = file['nrows'].values[0].astype(int)
    ncolumns = file['ncolumns'].values[0].astype(int)
    points_x = file['points x'].values[0:3]
    points_y = file['points y'].values[0:3]

    # Calculate spacing between points 1, 3, (nrows-1)
    space_x = points_x[2] - points_x[0]
    space_y = points_y[1] - points_y[0]

    # Generate coordinates and order them as layerprobe does
    coord_x = np.round(np.linspace(points_x[0], points_x[0] + space_x * (ncolumns - 1), ncolumns), 2)
    coord_y = np.round(np.linspace(points_y[0], points_y[0] + space_y * (nrows - 1), nrows), 2)

    X, Y = [], []
    for j in range(ncolumns):
        for i in range(nrows):
            Y.append(coord_y[i])
            X.append(coord_x[j])

    # Load the workbook and insert the calculated coordinates in the first sheet
    workbook = load_workbook(filepath)
    
    for i, value in enumerate(X, start=2):
        workbook['Sheet1'][f'B{i}'] = value
    for i, value in enumerate(Y, start=2):
        workbook['Sheet1'][f'C{i}'] = value

    workbook.save(newpath)
    workbook.close()
    
    print(filename, " - coordinates calculated and saved")

def EDX_sample_coords(folder, filename):
    'Calculate and translate EDX coordinates for a given file. Requires .xlsx file with columns as in template.'

    # Define file paths
    filepath = os.path.join(folder, filename+".xlsx")
    newpath = os.path.join(folder, filename+"_sample_coords.xlsx")

    # Read specific columns from the Excel file, and extract values
    file = pd.read_excel(filepath, sheet_name='Sheet2', usecols='H:P')
    file = file.drop(file.index[3:])  # Drop the 4th row (assumes you have info for 3 points)

    nrows= file['nrows'].values[0].astype(int)
    ncolumns = file['ncolumns'].values[0].astype(int)
    corners_x= file['corner x'].values[0:2]
    corners_y= file['corner y'].values[0:2]
    points_x = file['points x'].values[0:3]
    points_y = file['points y'].values[0:3]

    # Calculate spacing between points 1, 3, (nrows-1)
    space_x = points_x[2] - points_x[0]
    space_y = points_y[1] - points_y[0]

    # Calculate shift from corners and correct for this translation
    shift_x= (corners_x[1] +corners_x[0])/2
    shift_y= (corners_y[0] +corners_y[1])/2
    start_x = points_x[0] - shift_x
    start_y = points_y[0] - shift_y

    # Generate coordinates and order them as layerprobe does
    coord_x = np.round(np.linspace(start_x, start_x+ space_x*(ncolumns-1), ncolumns), 2)
    coord_y = np.round(np.linspace(start_y, start_y+ space_y*(nrows-1),    nrows), 2)
    X,Y = [],[]
    for j in range(0, ncolumns):
        for i in range(0, nrows):
            Y.append(coord_y[i])
            X.append(coord_x[j])

    # Load the workbook and insert the calculated coordinates in the first sheet
    workbook = load_workbook(filepath)
    sheet1= workbook['Sheet1']
    
    for i, value in enumerate(X, start= 2):
        sheet1[f'B{i}']= value
    for i, value in enumerate(Y, start= 2):
        sheet1[f'C{i}']= value

    workbook.save(newpath)
    workbook.close()

    # check for correct translation (allow 0.3 mm misalignment from sample rotation)
    if np.abs(X[-1]+X[0]) > 0.3 or np.abs(Y[-1]+Y[0]) > 0.3:
        print(filename, " - coordinates calculated and saved, but not symmetric")
        print("X shift: ", X[-1]+X[0])
        print("Y shift: ", Y[-1]+Y[0])
    else:
        print(filename, " - coordinates calculated, translated and saved")

def EDX_coordinates(folder, filename, edge=3, rotate=False, spacing= "auto"):
    'Calculate and translate EDX coordinates for a given file. Requires .xlsx file with columns as in template.'
    
    # Define file paths
    filepath = os.path.join(folder, filename+".xlsx")
    newpath = os.path.join(folder, filename+"_new_coords.xlsx")

    # Read specific columns from the Excel file, and extract values
    file = pd.read_excel(filepath, sheet_name='Sheet2', usecols='H:P')
    file = file.drop(file.index[3:])  # Drop the 4th row (assumes you have info for 3 points)

    nrows= file['nrows'].values[0].astype(int)
    ncolumns = file['ncolumns'].values[0].astype(int)
    corners_x= file['corner x'].values[0:2]
    corners_y= file['corner y'].values[0:2]
    mag = file['magnification'].values[0]
    if spacing == "auto":
        spacing = file['spacing'].values[0]

    if rotate == "90":
        ncolumns, nrows = nrows, ncolumns
        areax = 2.8 * 100 / mag
        areay = 4.1 * 100 / mag
    
    if rotate ==False:
        # Calculate the effective area size in the x-direction, considering magnification, 
        # assuming the x/y ratio is constant 4.1 : 2.8
        areax = 4.1 * 100 / mag
        areay= 2.8*100 / mag
    
    # Calculate the spacing , gridlength and starting x-coordinate for the grid in x-direction (assuming the grid is centered)
    space_x = areax * spacing / 100
    gridlength = (ncolumns - 1) * (space_x + areax) #+ areax
    startx = -gridlength / 2 #+ (areax / 2)

    # do the same for the y-direction
    
    space_y = areay*spacing/100
    gridheight= (nrows-1)*(space_y+ areay) #+ areay/2
    starty = -gridheight/2 #+ (areay/2)

    samplesize = [corners_x[1]-corners_x[0], corners_y[0]-corners_y[1]]
    print("Sample size is", samplesize)

    # Check if the grid dimensions exceed the maximum allowed size (31x31 mm)
    # if so, reduce the spacing by 10% and try again 
    if gridlength >= samplesize[0]-edge or gridheight >= samplesize[1]-edge:
        print("Spacing is too large for the map")

        new_spacing = np.round(spacing - spacing * 0.05, 0)
        print("New spacing is", new_spacing)

        return EDX_coordinates(folder, filename, spacing=new_spacing)

    # Generate coordinates for each column
    coord_x = np.round(np.linspace(startx, -startx, ncolumns), 2)
    coord_y = np.round(np.linspace(starty, -starty, nrows), 2)
    X=[]
    Y=[]
    for j in range(0, ncolumns):
        for i in range(0, nrows):
            Y.append(coord_y[i])
            X.append(coord_x[j])

    # Load the workbook and insert the calculated coordinates in the first sheet
    workbook = load_workbook(filepath)
    sheet1= workbook['Sheet1']
    
    for i, value in enumerate(X, start= 2):
        sheet1[f'B{i}']= value
    for i, value in enumerate(Y, start= 2):
        sheet1[f'C{i}']= value

    workbook.save(newpath)
    workbook.close()

def lp_translate_excel(folder,filename):
    """Creates a new excel file with translated coordinates, given the coordinates 
    of the corners in Sheet2, assuming they are stored rightafter the statistics"""
    filepath =os.path.join(folder,filename+".xlsx")
    newpath = os.path.join(folder,filename+"_translated.xlsx")

    first_data = pd.read_excel(filepath, sheet_name = "Sheet1")

    first_x = first_data["X (mm)"]
    first_y = first_data["Y (mm)"]

    corners = pd.read_excel(filepath, sheet_name = "Sheet2", usecols='K:L')

    trans_x = corners.iloc[[0,1],0].mean()
    trans_y = corners.iloc[[0,1],1].mean()

    new_x = first_x - trans_x
    new_y = first_y - trans_y

    new_data = first_data.copy()
    new_data["X (mm)"] = new_x
    new_data["Y (mm)"] = new_y

    # new_data.to_excel(new_path, index = False)

    workbook = load_workbook(filepath)
    sheet1= workbook['Sheet1']
    
    for i, value in enumerate(new_x, start= 2):
        sheet1[f'B{i}']= value
    for i, value in enumerate(new_y, start= 2):
        sheet1[f'C{i}']= value

    workbook.save(newpath)
    workbook.close()

def find_composition(data, el1,el2,el3, range1=[0,100], range2=[0,100], range3=[0,100], display_option=True, stoichiometry= None, tolerance= 3, sample='sample'):
    'find te points in the sample where the composition is in a certain range, given in % ranges or in stoichiometry and tolerance'

    if stoichiometry: 
        tot= sum(stoichiometry)
        range1= [(stoichiometry[0]*100/tot)-tolerance, (stoichiometry[0]*100/tot)+tolerance]
        range2= [(stoichiometry[1]*100/tot)-tolerance, (stoichiometry[1]*100/tot)+tolerance]
        range3= [(stoichiometry[2]*100/tot)-tolerance, (stoichiometry[2]*100/tot)+tolerance]

    ranges= [range1, range2, range3]
    elements= [el1, el2, el3]

    for i in range(0,len(elements)):
        idx_min= np.where(get_data(data, type= f'Layer 1 {elements[i]} Atomic %').values[0] >ranges[i][0])[0]
        idx_max= np.where(get_data(data, type= f'Layer 1 {elements[i]} Atomic %').values[0] <ranges[i][1])[0]
        idx= np.intersect1d(idx_max, idx_min)
        if i==0:
            idx1= idx
        elif i==1:
            idx2= idx
        elif i==2:
            idx3= idx
    idx= np.intersect1d(idx1, idx2) 
    idx= np.intersect1d(idx, idx3)
    x,y= extract_coordinates(data)
    good_comp= {'X': [], 'Y': []}
    for i in range(0,len(idx)):
        good_comp['X'].append(x[idx[i]])
        good_comp['Y'].append(y[idx[i]])

    good_comp= pd.DataFrame(good_comp)
    # display(good_comp)
    plt.scatter(good_comp['X'], good_comp['Y'], c='r', s=80)
    plt.scatter(x,y, c='b', s=10)
    plt.xlabel('x position (mm)')
    plt.ylabel('y position (mm)')
    if stoichiometry: 
        plt.title(f'{sample} - Positions where composition is {el1}{stoichiometry[0]}, {el2}{stoichiometry[1]}, {el3}{stoichiometry[2]} +-{tolerance}%')
    else:
        plt.title(f'{sample} - Positions where {el1}: {range1[0]:.1f}-{range1[1]:.1f}%, {el2}: {range2[0]:.1f}-{range2[1]:.1f}%, {el3}: {range3[0]:.1f}-{range3[1]:.1f}%')
    if display_option==True:
        for i in range(0,len(good_comp)):
            display(get_data(data, x= good_comp['X'][i], y= good_comp['Y'][i]))
    return good_comp

def calculate_ratio(df, el1, el2, rename= None):
    df= math_on_columns(df, f'Layer 1 {el1} Atomic %', f'Layer 1 {el2} Atomic %', "/")
    if rename:
        df.rename(columns={f'Layer 1 {el1} Atomic % / Layer 1 {el2} Atomic %': rename}, inplace=True)
    else:
        df.rename(columns={f'Layer 1 {el1} Atomic % / Layer 1 {el2} Atomic %': f'{el1}/{el2}'}, inplace=True)
    return df

def calculate_el_thickness(df, el):
    df= math_on_columns(df, f'Layer 1 {el} Atomic %', "Layer 1 Thickness (nm)", "*")
    df= math_on_columns(df, f'Layer 1 {el} Atomic % * Layer 1 Thickness (nm)',100, "/")
    df.rename(columns={f'Layer 1 {el} Atomic % * Layer 1 Thickness (nm) / 100': f'{el} [nm]'}, inplace=True)
    df.drop(columns=f'Layer 1 {el} Atomic % * Layer 1 Thickness (nm)', level=1, inplace=True)
    return df

def stats(data_all, type):
    data = get_data(data_all, type = type)
    data = data.sort_values(by = 0, axis=1 )
    min_= data.iloc[0,0]
    max_ = data.iloc[0,-1]
    mean_ = data.mean(axis=1)[0]

    data = pd.DataFrame([min_, max_, mean_], index = ["min","max", "mean"])
    return data

def rename_SE_images(folderpath):
    # Define the directory containing the images
    directory = folderpath  # Replace with the path to your folder

    # Function to extract the number from the filename
    def extract_number(filename):
        match = re.search(r'Electron Image (\d+)\.bmp', filename)
        return int(match.group(1)) if match else None

    # Get a list of all .bmp files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.bmp')]

    # Sort the files by their extracted number
    files.sort(key=extract_number)

    # Rename the files sequentially
    for i, filename in enumerate(files, start=1):
        new_name = f'Electron Image {i}.bmp'
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_name)
        
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} -> {new_name}')

    print("Renaming completed.")

def old_EDS_coordinates(ncolumns, nrows, mag, spacing, filepath, new_path, edge=4,rotate=False):
    """ 
    Function to generate EDS coordinates for a grid of a given size and magnification.
    Use this if the excel files do not have information as the template. 
    For files with the template, use EDX_sample_coords or EDX_stage_coords instead."""

    if rotate == "90":
        ncolumns, nrows = nrows, ncolumns
        areax = 2.8 * 100 / mag
        areay = 4.1 * 100 / mag
    
    if rotate ==False:
        # Calculate the effective area size in the x-direction, considering magnification, 
        # assuming the x/y ratio is constant 4.1 : 2.8
        areax = 4.1 * 100 / mag
        areay= 2.8*100 / mag
    
    # Calculate the spacing , gridlength and starting x-coordinate for the grid in x-direction (assuming the grid is centered)
    space_x = areax * spacing / 100
    gridlength = (ncolumns - 1) * (space_x + areax) + areax
    startx = -gridlength / 2 + (areax / 2)

    # do the same for the y-direction
    
    space_y = areay*spacing/100
    gridheight= (nrows-1)*(space_y+ areay) + areay
    starty = -gridheight/2 + (areay/2)

    # Check if the grid dimensions exceed the maximum allowed size (31x31 mm)
    # if so, reduce the spacing by 10% and try again 
    if gridlength >= 39-edge or gridheight >= 39-edge:
        print("Spacing is too large for the map")

        new_spacing = np.round(spacing - spacing * 0.1, 0)
        print("New spacing is", new_spacing)

        return EDS_coordinates(ncolumns, nrows, mag, new_spacing, filepath, new_path,rotate)

    # Create a list to hold grid parameters (input of the grid function)
    grid_input = [ncolumns, nrows,np.round(-startx*2, 2), np.round(-starty*2, 2), np.round(startx, 2), np.round(starty, 2)]

    # Generate coordinates for each column
    coord_x = np.round(np.linspace(startx, -startx, ncolumns), 2)
    coord_y = np.round(np.linspace(starty, -starty, nrows), 2)
    X=[]
    Y=[]
    for j in range(0, ncolumns):
        for i in range(0, nrows):
            Y.append(coord_y[i])
            X.append(coord_x[j])

    first_data = pd.read_excel(filepath, sheet_name = "Sheet1")

    new_data = first_data.copy()
    new_data["X (mm)"] = X
    new_data["Y (mm)"] = Y

    new_data.to_excel(new_path, index = False)

    return X,Y, grid_input, areax, areay
