


# NOTE: Taken from https://github.com/DTU-Nanolab-materials-discovery/intern-data-handling-and-fitting
# All credits to https://github.com/giul-s

# === Functions for XRD analysis ===

def XRD_background(data,peaks, cut_range=2, order=4,window_length=10, Si_cut=True, withplots= True ):
    data_out = data.copy()
    headerlength = len(data.columns.get_level_values(1).unique())
    col_theta = data.columns.values[::2]
    col_counts = data.columns.values[1::2]
    peaks_theta = peaks.columns.values[::2]

    k=0

    for i in range(0, len(col_theta)):
        cut_intensity=[]

        two_theta = data[col_theta[i]]
        intensity = data[col_counts[i]]
        idx_range = np.where(two_theta >= 20+cut_range)[0][0]

        # Cut data around peaks
        for j in range(len(intensity)):
            if data[col_theta[i]][j] in peaks[peaks_theta[i]].values:
                start_index = max(0, j-idx_range)
                end_index = min(len(data), j+idx_range)
                data_out[col_counts[i]][start_index:end_index] = np.nan #cut data intensity around peaks in data_out

        if Si_cut==True:
            idx_Si = np.where((two_theta >= 60) & (two_theta<= 70))[0]
            data_out[col_counts[i]][idx_Si] = np.nan

        cut_intensity = data_out[col_counts[i]]

        # Smooth the data for better peak detection
        smoothed_intensity = savgol_filter(intensity, window_length=window_length, polyorder=3)
        # Filter out NaN values (they exist because we cut the data) before fitting
        mask = ~np.isnan(cut_intensity)
        filtered_two_theta = two_theta[mask]
        filtered_intensity = intensity[mask]

        # Perform polynomial fitting with filtered data
        background_poly_coeffs = np.polyfit(filtered_two_theta, filtered_intensity, order)
        background = np.polyval(background_poly_coeffs, two_theta)

        # Subtract background
        corrected_intensity = smoothed_intensity - background

        data_out.insert(headerlength*(i+1)+k, "{}".format(data.columns.get_level_values(0).unique()[i]), background, allow_duplicates=True)
        data_out.rename(columns={'': 'Background'}, inplace = True)
        data_out.insert(headerlength*(i+1)+k+1, "{}".format(data.columns.get_level_values(0).unique()[i]), corrected_intensity, allow_duplicates=True)
        data_out.rename(columns={'': 'Corrected Intensity'}, inplace = True)
        k=k+2

        if withplots==True:
            plt.figure()
            coord= data_out.columns.get_level_values(0).unique()[i]
            plt.plot(two_theta, intensity, label='Original Data')
            plt.plot(filtered_two_theta, filtered_intensity, label='filtered Data')
            plt.plot(two_theta, background, label='Background, order='+str(order), linestyle='--')
            plt.plot(two_theta, corrected_intensity, label='Corrected Data')
            plt.title('XRD data at {}'.format(coord))
            plt.legend()
            plt.show()
    display(data_out)
    
    return data_out

def initial_peaks(data,dataRangeMin, dataRangeMax,filterstrength, peakprominence,peakwidth, peakheight=0, withplots = True, plotscale = 'log'):
    '''finds peaks using scipy find_peaks on filtered data to construct a model for fitting, filter strength is based on filtervalue and 
    peak find sensitivity based on peakprominence, withplots and plotscale allows toggling plots on/off and picking scale.
    Output: dataframe with peak locations and intensity, to be used for raman_fit or xrd_fit, and data limited by the dataRangemin/max, in index'''
    #setup data
    column_headers = data.columns.values
    col_theta = column_headers[::2]
    col_counts = column_headers[1::2]
    data = data.iloc[dataRangeMin:dataRangeMax]
    data.reset_index(drop=True, inplace=True)

    #create list for intital peaks
    thePeakss = []
    dataCorrect1 = []

    #finding the peaks
    for i in range(0,len(col_theta)):
        #select data
        dataSelect = data[col_counts[i]].copy()
        x = data[col_theta[i]]
        

        #Filter to avoid fake peaks
        if filterstrength > 0:
            l = filterstrength
            #dataSelect = lfilter(b, a, dataSelect)
            dataSelect = savgol_filter(dataSelect, filterstrength, 1)
        

        #find peaks
        peaks, _ = find_peaks(dataSelect,height=peakheight,prominence= peakprominence,width = peakwidth)

        #plot
        if withplots == 1:
            plt.plot(x,dataSelect)
            plt.plot(x[peaks], dataSelect[peaks], 'x')
            plt.yscale(plotscale)
            plt.xlabel(col_theta[i][1])
            plt.ylabel(col_counts[i][1])
            plt.title(col_counts[i][0])
            plt.show()


        #save peaks data in the list
        peaksOut = data[[col_theta[i], col_counts[i]]].loc[peaks]
        peaksOut.reset_index(drop=True, inplace=True)
        thePeakss.append(peaksOut)

        #save peaks data in the list
        dataCorr = np.vstack((x,dataSelect)).T
        #dataCorr = pd.DataFrame(data=dataCorrect, columns=column_headers)
        dataCorrect1.append(dataCorr)

    #convert list to dataframe
    thePeaks = pd.concat(thePeakss, axis=1)
    dataCorrected = np.concatenate(dataCorrect1, axis=1)
    dataCorrected = pd.DataFrame(dataCorrected, columns = data.columns)
    return thePeaks, dataCorrected

def plot_XRD_shift_subplots(data, datatype_x, datatype_y, x, y_list, shift, title, material_guess, nrows, ncols, figsize=(12, 10), save=True):
    """
    Plots XRD shift for multiple y-coordinates in subplots. 
    """
    with open (os.path.join("XRD", "reflections", "reflections.pkl"), "rb") as file:
        ref_peaks_df = pickle.load(file)

    ref_peaks = ref_peaks_df[material_guess]
    ref_lines = ref_peaks["Peak 2theta"][ref_peaks["Peak 2theta"].notna()].values
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, pos in enumerate(y_list):
        ax = axes[idx]
        for i in range(len(x)):
            #print('x =', x[i], 'y =', pos[i])
            x_data = get_data(data, datatype_x, x[i], pos[i], printinfo=False, drop_nan=False)
            y_data = get_data(data, datatype_y, x[i], pos[i], printinfo=False, drop_nan=False)
            lab = "{:.1f},{:.1f}".format(x[i], pos[i])

            ax.plot(x_data, y_data + shift * i, label=lab)

        ax.set_title(f'Y = {pos[0]}')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        if ref_lines is not None:
            for line in ref_lines:
                ax.axvline(x=line, linestyle='--', alpha=0.5, color='grey')


    axes[-1].plot(ref_peaks["2theta"], ref_peaks["I"], label=str(material_guess)) 
    #axes[-1].axvline(x=ref_lines.values, linestyle='--', alpha=0.5, color='grey')      
    axes[-1].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        plt.savefig(f'{title}_XRD_shift_subplots.png', dpi=120, bbox_inches='tight')

    plt.show()

def plot_XRD_shift(data,datatype_x, datatype_y,  shift,x,y, title=None, savepath= False, show=True): #x, y = list of points to plot]
    x_data = []
    y_data = []
    labels = []
    plt.figure(figsize = (12,5))
    for i in range(len(x)):
        x_data.append(get_data(data, datatype_x, x[i], y[i], False,False))
        y_data.append(get_data(data, datatype_y, x[i], y[i], False,False))
        if x[0] == "all" and y[0] == "all":
            labels = data.columns.get_level_values(0).unique().values
        else:
            grid = MI_to_grid(data)
            xcoord, ycoord = closest_coord(grid, x[i], y[i])
            labels.append('{:.1f},{:.1f}'.format(xcoord, ycoord))

        plt.plot(x_data[i], y_data[i]+ shift*i, label = labels[i])
    plt.xlabel(datatype_x)
    plt.ylabel(datatype_y)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if savepath:
        path = os.path.join('plots', title + 'shift.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
    
    if show==True:
              plt.show()

def fit_two_related_peaks(x, y):
    """ Makes model with two related peaks and fits it to the data. 
    Returns the fit result and the fitted parameters for both peaks. """

    # Initialize two Pseudo-Voigt models with prefixes to distinguish parameters
    model1 = PseudoVoigtModel(prefix='p1_')
    model2 = PseudoVoigtModel(prefix='p2_')

    # Estimate initial parameters for the first peak
    params = model1.guess(y, x=x)
    
    # Extract initial guesses
    amplitude1 = params['p1_amplitude'].value
    center1 = params['p1_center'].value
    sigma1 = params['p1_sigma'].value
    fraction1 = params['p1_fraction'].value

    # Set constraints for the second peak based on the provided relations
    #xpeak2 = 2 * np.arcsin((0.154439 / 0.1540562) * np.sin(center1 / 2))
    xpeak2= (360/np.pi)* np.arcsin((0.154439 / 0.1540562) * np.sin(center1*np.pi /360))
    
    params.add('p2_center', expr='(360/pi)* arcsin((0.154439 / 0.1540562) * sin(p1_center*pi /360))')
    params.add('p2_amplitude', expr='0.5 * p1_amplitude')
    params.add('p2_sigma', expr='1 * p1_sigma')
    params.add('p2_fraction', expr='1 * p1_fraction')

    # Create a combined model by summing the two models
    combined_model = model1 + model2

    # Perform the fit
    fit_result = combined_model.fit(y, params, x=x)

    # Extract the fitted parameters for both peaks
    amplitude1 = fit_result.params['p1_amplitude'].value
    center1 = fit_result.params['p1_center'].value
    sigma1 = fit_result.params['p1_sigma'].value
    fraction1 = fit_result.params['p1_fraction'].value

    amplitude2 = fit_result.params['p2_amplitude'].value
    center2 = fit_result.params['p2_center'].value
    sigma2 = fit_result.params['p2_sigma'].value
    fraction2 = fit_result.params['p2_fraction'].value

    # Calculate FWHM for both peaks
    gamma1 = sigma1 / np.sqrt(2 * np.log(2))  # Convert sigma to gamma for Gaussian part
    fwhm1 = (1 - fraction1) * (2 * gamma1) + fraction1 * (2 * sigma1)

    gamma2 = sigma2 / np.sqrt(2 * np.log(2))
    fwhm2 = (1 - fraction2) * (2 * gamma2) + fraction2 * (2 * sigma2)

    return fit_result, amplitude1, fwhm1, center1, fraction1, amplitude2, fwhm2, center2, fraction2

def fit_this_peak(data, peak_position, fit_range, withplots = True, printinfo = False):
    """ Given a peak position, fits a double Pseudo-Voigt model to the data around the peak."""
    cut_range = fit_range
    peak_angle = peak_position

    dat_theta = data.iloc[:,data.columns.get_level_values(1)=='2θ (°)']
    dat_counts = data.iloc[:,data.columns.get_level_values(1)=='Corrected Intensity']

    colors = plt.cm.jet(np.linspace(0, 1, len(dat_theta.columns)))

    plt.figure(figsize=(8, 6))

    df_fitted_peak = pd.DataFrame()

    for i in range(0, len(dat_theta.columns)):
        data_to_fit_x = dat_theta[dat_theta.columns[i]]
        data_to_fit_y = dat_counts[dat_counts.columns[i]]

        idx = np.where((data_to_fit_x >= peak_angle-cut_range) & (data_to_fit_x<=  peak_angle+cut_range))[0]
        x_range = data_to_fit_x[idx].values
        y_range = data_to_fit_y[idx].values

        fit_result, amplitude1, fwhm1, center1, fraction1, amplitude2, fwhm2, center2, fraction2 = fit_two_related_peaks(x_range, y_range)

        if printinfo == True:
            print(dat_theta.columns[i][0])
            print(f"Peak 1 - Amplitude: {amplitude1:.2f}, FWHM: {fwhm1:.2f}, Center: {center1:.2f}, Fraction: {fraction1:.2f}")
            print(f"Peak 2 - Amplitude: {amplitude2:.2f}, FWHM: {fwhm2:.2f}, Center: {center2:.2f}, Fraction: {fraction2:.2f}")

        if withplots==True:
            plt.plot(x_range, y_range, 'o', color = colors[i], label=str(dat_theta.columns[i][0]))
            plt.plot(x_range, fit_result.best_fit, '-', color = colors[i])
            plt.xlabel('2θ')
            plt.ylabel('Intensity')
            plt.title(' Fit with two related PseudoVoigts at '+ str(peak_angle) + '°')

        # store the information about the peak in a new dataframe 

        peakData = np.vstack((center1, amplitude1, fwhm1,  fraction1)).T
        peak_header = pd.MultiIndex.from_product([[dat_theta.columns[i][0]], ["Center","Amplitude", "FWHM", "Fraction"]], names = ["Coordinate", "Data type"])
        df_peak_info=pd.DataFrame( data= peakData, columns = peak_header)
        fitData = np.vstack((x_range, y_range, fit_result.best_fit)).T
        fit_header = pd.MultiIndex.from_product([[dat_theta.columns[i][0]], ["range 2θ","range Intensity", "Fit"]], names = ["Coordinate", "Data type"])
        df_fit_info = pd.DataFrame(data = fitData, columns = fit_header)
        df_fitted_peak = pd.concat([df_fitted_peak, df_fit_info, df_peak_info], axis=1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    display(df_fitted_peak)
    return df_fitted_peak

def rgba_to_hex(rgba):
    """Convert an RGBA tuple to a hex color string."""
    r, g, b, a = [int(c * 255) for c in rgba]
    return f'#{r:02x}{g:02x}{b:02x}'

def interactive_XRD_shift(data, datatype_x, datatype_y, shift, x, y, ref_peaks_df, title=None, colors='rows', savepath=None):
    'interactive shifted plot for assigning phases to XRD data, specify if you want different colors per each row or a rainbow colormap'
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        row_heights=[0.8, 0.2],  # Proportion of height for each plot
        vertical_spacing=0.02    # Adjust this to reduce space between plots
    )
    
    if colors == 'rows':
        # Define a color palette with as many colors as there are unique values in y
        coords_colors = pd.DataFrame({'X': x, 'Y': y})
        unique_y_values = coords_colors['Y'].unique()
        
        color_palette = px.colors.qualitative.G10[:len(unique_y_values)]
        
        unique_x_values = coords_colors['X'].unique()
        color_dict = {}
        for i, color in enumerate(color_palette):
            # Generate lighter hues of the color for each x value
            base_color = mcolors.to_rgb(color)
            lighter_hues = [mcolors.to_hex((base_color[0] + (1 - base_color[0]) * (j / len(unique_x_values)),
                                            base_color[1] + (1 - base_color[1]) * (j / len(unique_x_values)),
                                            base_color[2] + (1 - base_color[2]) * (j / len(unique_x_values))))
                            for j in range(len(unique_x_values))]
            color_dict[unique_y_values[i]] = lighter_hues
        coords_colors['Color'] = coords_colors.apply(lambda row: color_dict[row['Y']][list(unique_x_values).index(row['X'])], axis=1)
        colors = coords_colors['Color'].values

    elif colors == 'rainbow':
        colormap = plt.get_cmap('turbo')  # You can choose any matplotlib colormap
        colors = [rgba_to_hex(colormap(i / len(x))) for i in range(len(x))]  # Convert colors to hex
    
    x_data = []
    y_data = []
    # Store all y-data to find the global maximum
    all_y_data = []
    # Loop through and plot the XRD spectra with a vertical shift in the top plot
    for i in range(len(x)):
        x_data = get_data(data, datatype_x, x[i], y[i], False, False)
        y_data = get_data(data, datatype_y, x[i], y[i], False, False)
        shifted_y_data = y_data - shift * i
        
        all_y_data.extend(shifted_y_data)  # Collect y-data with shift for max computation
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=shifted_y_data,
                mode='lines',
                line=dict(color=colors[i]),
                name=f'{i+1}: {x[i]}, {y[i]}'
            ),
            row=1, col=1
        )

    # Compute the global maximum y-value, considering shifts
    global_min_y = min(all_y_data)

    # Create traces for each reference material (hidden initially)
    ref_traces = []
    buttons = []

    for ref_material, ref_df in ref_peaks_df.items():
        # Reference spectrum plotted in the bottom plot
        ref_trace = go.Scatter(
            x=ref_df["2theta"],
            y=ref_df["I"],
            mode='lines',
            name=f'{ref_material} Reference',
            visible=False
        )
        
        # Create vertical peak lines for top plot (raw data plot)
        peak_lines = go.Scatter(
            x=[value for peak in ref_df["Peak 2theta"] for value in [peak, peak, None]],  # x: peak, peak, None to break the line
            y=[global_min_y, 1000 * 1.1, None] * len(ref_df["Peak 2theta"]),  # y: 0 -> global_max_y for each line, with None to break lines
            mode='lines',
            line=dict(color='grey', dash='dot'),
            showlegend=False,
            visible=False
        )

        # Append traces for each reference spectrum and its peaks
        ref_traces.append(ref_trace)
        ref_traces.append(peak_lines)
        
        # Create a button for each reference
        buttons.append(dict(
            label=ref_material,
            method='update',
            args=[{'visible': [True] * len(x) + [False] * len(ref_traces)},  # Show all raw spectra, hide refs by default
                  {'title': f'{title} - {ref_material} Reference'}]
        ))

    # Add reference traces to figure (initially hidden)
    for trace in ref_traces:
        # Ensure trace.name is not None before checking 'Reference' in name
        fig.add_trace(trace, row=2 if trace.name and 'Reference' in trace.name else 1, col=1)

    # Update buttons to control the visibility of one reference at a time
    for i, button in enumerate(buttons):
        # Make the selected reference spectrum visible in the bottom plot and its peaks visible in the top plot
        button['args'][0]['visible'][len(x):] = [False] * len(ref_traces)  # Hide all refs initially
        button['args'][0]['visible'][len(x) + 2 * i:len(x) + 2 * i + 2] = [True, True]  # Show selected ref and peaks

    # Add the dropdown menu to switch between reference spectra
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }],
        template='plotly_white',  # Choose a template (e.g., 'plotly_dark')
        title=title,
        height=600,  # Adjust the height of the figure (e.g., 700)
        width=900,   # Adjust the width of the figure (e.g., 900)
        legend=dict(x=1.05, y=1),
        xaxis2_title=datatype_x,
        yaxis_title=datatype_y
    )

    if savepath:
        fig.write_html(savepath)
        
    fig.show()
    return fig

def assign_phases_labels(data):
    """Function to assign phases to specific points in a dataset.
    Returns:
        phase_info (dict): Dictionary where the key is the phase and the value is a list of 'unknown', 'amorphous', or the phase name
                           corresponding to the presence of that phase at each coordinate.
    """
    coords = data.columns.get_level_values(0).unique()

    phase_info = {}  # Dictionary to store phase information for each point
    num_coords = len(coords)
    
    # Initialize the presence array with 'unknown'
    phase_present = ["unknown"] * num_coords

    # Ask user for the main phase name
    main_phase = input("What is the main phase present? (or type 'exit' to finish): ").strip()
    if main_phase.lower() == 'exit':
        phase_info["Phase"] = phase_present
        return phase_info  # Return the dictionary with 'unknown' if user exits

    # Assign the main phase to all points initially
    phase_present = [main_phase] * num_coords

    while True:
        # Ask if there is any other phase
        other_phase_response = input("Is there any other phase present? (yes/no): ").strip().lower()
        if other_phase_response == 'no':
            break

        # Ask user for the other phase name
        other_phase = input("What is the other phase name? (or type 'exit' to finish): ").strip()
        if other_phase.lower() == 'exit':
            break

        # Display available points
        print("\nAvailable points (coordinates):")
        for i, coord in enumerate(coords):
            print(f"{i + 1}: {coord}")

        # Ask which points should be set to the other phase
        selected_points = input(f"\nWhich points should be set to '{other_phase}'? (Enter numbers separated by commas): ").strip()
        selected_indices = [int(idx.strip()) - 1 for idx in selected_points.split(',') if idx.strip().isdigit()]
        for idx in selected_indices:
            if 0 <= idx < num_coords:
                phase_present[idx] = other_phase

    # Store this phase's information in the dictionary
    phase_info["Phase"] = phase_present
    print(f"\nPhase '{main_phase}' assigned to the remaining points.")
    print(phase_info)
    
    return phase_info

def assign_phases_numbers(data):
    
    """obsolete, use phase labels instead.
    Function to assign phases to specific points in a dataset.  coords (list of tuples): List of coordinates available for selection.
    Returns:
        phase_info (dict): Dictionary where the key is the phase and the value is a list of 'yes'/'no'
                           corresponding to the presence of that phase at each coordinate.
    """
    coords= data.columns.get_level_values(0).unique()

    phase_info = {}  # Dictionary to store phase information for each point
    num_coords = len(coords)
    
    # Ask user for the phase name
    phase = input("What is the phase name? (or type 'exit' to finish): ").strip()
    if phase.lower() == 'exit':
        return phase_info  # Return empty dictionary if user exits

    # Determine if phase is present in most or few points
    presence_type = input("Is the phase present in most points or few points? (type 'most' or 'few'): ").strip().lower()

    # Initialize the presence array based on user input
    if presence_type == 'most':
        phase_present = [1] * num_coords  # Start with all points as 'yes'
    elif presence_type == 'few':
        phase_present = [0] * num_coords  # Start with all points as 'no'
    else:
        print("Invalid input. Please enter 'most' or 'few'.")

    # Display available points
    print("\nAvailable points (coordinates):")
    for i, coord in enumerate(coords):
        print(f"{i + 1}: {coord}")

    # Ask which points should be changed
    if presence_type == 'most':
        selected_points = input(f"\nWhich points should be set to 'no'? (Enter numbers separated by commas): ").strip()
        selected_indices = [int(idx.strip()) - 1 for idx in selected_points.split(',') if idx.strip().isdigit()]
        for idx in selected_indices:
            if 0 <= idx < num_coords:
                phase_present[idx] = 0
    else:  # presence_type == 'few'
        selected_points = input(f"\nWhich points should be set to 'yes'? (Enter numbers separated by commas): ").strip()
        selected_indices = [int(idx.strip()) - 1 for idx in selected_points.split(',') if idx.strip().isdigit()]
        for idx in selected_indices:
            if 0 <= idx < num_coords:
                phase_present[idx] = 1

    # Store this phase's information in the dictionary
    phase_info[phase] = phase_present
    print(f"\nPhase '{phase}' assigned to the selected points.")
    print(phase_info)
    
    return phase_info


## Malthe and Rasmus functions for XRD

def make_model_xrd(num,i,peaks,col_counts,col_theta,params):
    '''Obsolete: used in xrd_fit. 
    Constructs a pseudovoigt model for every peak based on peaks output from initial_peaks'''
    pref = "f{0}_".format(num)
    model = PseudoVoigtModel(prefix=pref)
    ypeak = peaks[col_counts[i]][peaks.index[num]]
    xpeak = peaks[col_theta[i]][num].astype(float)
    #width = widths_initial[num]
    params.update(model.make_params(center = dict(value=xpeak, min=xpeak*0.9, max=xpeak*1.1),
                                    amplitude = dict(value=ypeak, min=0.2 * ypeak, max=1.2*ypeak)
                                    ))
    return model

def xrd_fit(data,Peaks,dataRangeMin, dataRangeMax,knots, withplots = True,plotscale = 'log',remove_background_fit = False):
    ''' Obsolete: Fits the whole pattern, spline background can cause artefacts. Fit one peak at a time using fit_this_peak
        Fit data using models from lmfit. Pseudo-voigt for peaks, based on thePeaks output from initial_peaks, and 
    spline background model adjustable with knots. withplots and plotscale allows toggling plots on/off and picking scale. 
    Outputs: dataframe with theta, measured intensity, the entire fit, peak locations, intensity, FWHM, and Lorentzian/Gaussian fraction '''
    #setup data
    column_headers = data.columns.values
    col_theta = column_headers[::2]
    col_counts = column_headers[1::2]
    data = data.iloc[dataRangeMin:dataRangeMax]
    data.reset_index(drop=True, inplace=True)
    
    #empty frame for XRD output
    XRDoutFrame = pd.DataFrame()

    #fit all the things number 2
    for i in range(0,len(col_theta)):
        #select data
        x = data[col_theta[i]]
        y = data[col_counts[i]]

        #select peaks and remove nans
        thesePeaks = Peaks[[col_theta[i],col_counts[i]]].dropna()

        #define peak model
        mod = None
        peakNames = []
        params = Parameters()
        for l in range(len(thesePeaks)):
            this_mod = make_model_xrd(l,i,thesePeaks,col_counts,col_theta,params)
            if mod is None:
                mod = this_mod
            else:
                mod = mod + this_mod
            peakNames.append(this_mod.prefix)


        #define background model
        knot_xvals = np.linspace(min(x), max(x), knots)
        bkg = SplineModel(prefix='bkg_', xknots = knot_xvals)
        params = params.update(bkg.guess(y,x=x))

        #construct model
        mod = mod + bkg

        #fit
        out = mod.fit(y, params, x=x)
        comps = out.eval_components(x=x)    

        #extract peak data from fit
        peakHeights = np.array([])
        peakCenters = np.array([])
        peakFWHMs = np.array([])
        peakFractions = np.array([])
        for j in range(len(peakNames)):
            peakCenter = round(out.params[peakNames[j] + 'center'].value,2)
            peakHeight  = round(out.params[peakNames[j] + 'height'].value,3)
            peakHeights = np.append(peakHeights,peakHeight)
            peakCenters = np.append(peakCenters,peakCenter)
            peakFWHMs = np.append(peakFWHMs,round(out.params[peakNames[j] + 'fwhm'].value,2))
            peakFractions = np.append(peakFractions,round(out.params[peakNames[j] + 'fraction'].value,2))

        peakData = np.vstack((peakCenters,peakHeights,peakFWHMs,peakFractions)).T
        XRD_peaks_header = pd.MultiIndex.from_product([[col_theta[i][0]],['Peak 2θ','Peak intensity','FWHM','Lorentzian/Gaussian fraction']],names=['Coordinate','Data type'])
        columns = ['Peak 2θ','Peak intensity','FWHM','Lorentzian/Gaussian fraction']
        peakOutput = pd.DataFrame(data=peakData, columns=XRD_peaks_header)

        #extract fit and theta
        XRD_data_header = pd.MultiIndex.from_product([[col_theta[i][0]],['2θ','Measured intensity','Fit intensity', 'Background']],names=['Coordinate','Data type'])
        if remove_background_fit != False:
            fitData = np.vstack((x,y,out.best_fit-comps['bkg_'], comps['bkg_'])).T
        else:
            fitData = np.vstack((x,y,out.best_fit, comps['bkg_'])).T
        fitOutput = pd.DataFrame(data=fitData, columns=XRD_data_header)
        XRDoutFrame = pd.concat([XRDoutFrame, fitOutput, peakOutput], axis = 1)

        #plot fit
        if withplots == 1:
            plt.plot(x,y, label = 'data')
            plt.plot(x, out.best_fit, label='best fit')
            plt.plot(x, comps['bkg_'],'--', label='background')
            plt.xlabel(col_theta[i][1])
            plt.ylabel(col_counts[i][1])
            plt.title(col_counts[i][0])
            plt.yscale(plotscale)
            plt.legend()
            plt.show()
            #print output peak positions, intensity, and FWHM
            print("Peak positions:\n",peakOutput)       


    return XRDoutFrame
