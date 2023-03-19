""" Working script to analyze Discrete NIR Data """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal 
from lib.postprocess import SpectraGen

# wavelengths
xs = [855, 940, 1050, 1200, 1300, 1450, 1550, 1650]

np.set_printoptions(precision=6, suppress=True)

def plot1(save=False):
    """
    Plot comparing ABS to PLA of the same colour. 
    """

    # Read data
    df1 = pd.read_csv("./data/pla_red_1d_no_colar.csv")
    df2 = pd.read_csv("./data/abs_red_1n_no_colar.csv")
    df1 = df1.drop(['Unnamed: 0'], axis=1)
    df2 = df2.drop(['Unnamed: 0'], axis=1)
    print(df1.head())

    # means
    mean1 = df1.mean(axis=0).values
    mean2 = df2.mean(axis=0).values
    print(mean1)

    fig1, (ax1) = plt.subplots(1, 1, figsize=(12,7), sharex=True)

    # Means & Best plot
    ax1.set_title(f"DNIR Reading: ABS vs PLA (same colour)")

    std1 = []
    std2 = []
    for i in range(0, 8):
        values1 = df1[f'led{i+1}'].values
        values2 = df2[f'led{i+1}'].values
        std1.append(np.std(values1))
        std2.append(np.std(values2))

        ax1.scatter([xs[i]]*len(df1[f'led{i+1}'].values), values1, c='m')
        ax1.scatter([xs[i]]*len(df2[f'led{i+1}'].values), values2, c='c')

    # Add mean lines
    ax1.plot(xs, mean1, c='m', label='pla_mean', linestyle='--')
    ax1.plot(xs, mean2, c='c', label='abs_mean', linestyle='--')
    
    mean_std1 = np.format_float_scientific(np.asarray(std1).mean(axis=0), precision=2)
    mean_std2 = np.format_float_scientific(np.asarray(std2).mean(axis=0), precision=2)

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend([f"pla [std: {mean_std1}]", f"abs [std: {mean_std2}]"])

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"DNIR_abs_pla_red_plot1"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot2(save=False):
    """
    Compare discrete and continous ABS / PLA spectra. 
    """

    # Read data
    df1 = pd.read_csv("./data/pla_red_1d_no_colar.csv")
    df2 = pd.read_csv("./data/abs_red_1n_no_colar.csv")
    df1 = df1.drop(['Unnamed: 0'], axis=1)
    df2 = df2.drop(['Unnamed: 0'], axis=1)
    # Map any none-zero values to zero
    df1[df1 < 0] = 0.0
    # Map any none-zero values to zero
    df2[df2 < 0] = 0.0
    print(df1.head())

    # min / max scaling
    min1 = min(df1.to_numpy().min(), df2.to_numpy().min())
    max1 = max(df1.to_numpy().max(), df2.to_numpy().max())
    print(f"DNIR | min: {min1}, max: {max1}")

    # means    
    mean1 = ( df1.mean(axis=0).values - min1) / (max1 - min1)
    mean2 = ( df2.mean(axis=0).values - min1) / (max1 - min1)

    # Read data
    df3 = pd.read_csv("./data/pla_ftir_scan.csv", names=['x', 'y'])
    df4 = pd.read_csv("./data/abs_ftir_scan.csv", names=['x', 'y'])

    min2 = min(df3['y'].min(axis=0), df4['y'].min(axis=0))
    max2 = max(df3['y'].max(axis=0), df4['y'].max(axis=0))
    print(f"FTIR | min: {min2}, max: {max2}")

    # apply min / max scaling
    df3['y'] = ( df3['y'] - min2) / (max2 - min2)
    df4['y'] = ( df4['y'] - min2) / (max2 - min2)

    fig1, (ax1) = plt.subplots(1, 1, figsize=(12,7), sharex=True)

    # Means & Best plot
    ax1.set_title(f"DNIR vs FTIR Reading: (ABS vs PLA)")

    # DNIR
    ax1.scatter(xs, mean1, c='m')
    ax1.scatter(xs, mean2, c='c')

    # FTIR
    ax1.plot(df3['x'], df3['y'], c='r')
    ax1.plot(df4['x'], df4['y'], c='b')

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend([f"pla_DNIR", f"abs_DNIR", "pla_FTIR", "abs_FTIR"])

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"DNIR_v_FTIR_plot1"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot3(save=False):
    """
    Continous ABS / PLA spectra. 
    """

    # Read data
    df1 = pd.read_csv("./data/pla_ftir_scan.csv", names=['x', 'y'])
    df2 = pd.read_csv("./data/abs_ftir_scan.csv", names=['x', 'y'])
    print(df1.head())

    fig1, (ax1) = plt.subplots(1, 1, figsize=(12,7), sharex=True)

    # Means & Best plot
    ax1.set_title(f"FTIR Reading: ABS vs PLA")

    # Add mean lines
    ax1.plot(df1['x'], df1['y'], c='m')
    ax1.plot(df2['x'], df2['y'], c='c')

    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend([f"pla", f"abs"])
    # plt.grid()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"FTIR_plot1"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot4(save=False):
    """
    Continous ABS / PLA spectra with discrete points of interest. 
    """

    # Read data
    df1 = pd.read_csv("./data/pla_ftir_scan.csv", names=['x', 'y'])
    df2 = pd.read_csv("./data/abs_ftir_scan.csv", names=['x', 'y'])
    print(df1.info())

    fig1, (ax1) = plt.subplots(1, 1, figsize=(12,7), sharex=True)

    # Means & Best plot
    ax1.set_title(f"Continous ABS vs PLA with discrete measurement points")

    # Add mean lines
    ax1.plot(df1['x'][:2500], df1['y'][:2500], c='m')
    ax1.plot(df2['x'][:2500], df2['y'][:2500], c='c')

    for i in range(0, 8):
        plt.axvline(x = xs[i], color = 'k', linestyle='--')

    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend([f"pla", f"abs", "discrete"])

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"FTIR_highlighted_plot1"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot5(save=False):
    """
    Compare discrete and continous ABS / PLA spectra. 
    """

    # Read data
    df1 = pd.read_csv("./data/pla_red_1d_no_colar.csv")
    df2 = pd.read_csv("./data/abs_red_1n_no_colar.csv")
    df3 = pd.read_csv("./data/pla_ftir_scan.csv", names=['x', 'y'])
    df4 = pd.read_csv("./data/abs_ftir_scan.csv", names=['x', 'y'])

    df1 = df1.drop(['Unnamed: 0'], axis=1)
    df2 = df2.drop(['Unnamed: 0'], axis=1)
    # Map any none-zero values to zero
    df1[df1 < 0] = 0.0
    # Map any none-zero values to zero
    df2[df2 < 0] = 0.0
    print(df1.head())

    # min / max scaling
    min1 = min(df1.to_numpy().min(), df2.to_numpy().min())
    max1 = max(df1.to_numpy().max(), df2.to_numpy().max())
    min2 = min(df3['y'].min(axis=0), df4['y'].min(axis=0))
    max2 = max(df3['y'].max(axis=0), df4['y'].max(axis=0))
    print(f"DNIR | min: {min1}, max: {max1}")
    print(f"FTIR | min: {min2}, max: {max2}")

    # means    
    mean1 = np.asarray(( df1.mean(axis=0).values - min1) / (max1 - min1))
    mean2 = np.asarray(( df2.mean(axis=0).values - min1) / (max1 - min1))

    # apply min / max scaling
    df3['y'] = ( df3['y'] - min2) / (max2 - min2)
    df4['y'] = ( df4['y'] - min2) / (max2 - min2)

    fig1, (ax1) = plt.subplots(1, 1, figsize=(12,7), sharex=True)

    # Means & Best plot
    ax1.set_title(f"DNIR vs FTIR Reading: (ABS vs PLA)")

    # DNIR
    ax1.scatter(xs, mean1, c='m')
    ax1.scatter(xs, mean2, c='c')

    # FTIR
    ax1.plot(df3['x'][:2500], df3['y'][:2500], c='m')
    ax1.plot(df4['x'][:2500], df4['y'][:2500], c='c')

    for i in range(0, 8):
        plt.axvline(x = xs[i], color = 'k', linestyle='--')

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend([f"pla_DNIR", f"abs_DNIR", "pla_FTIR", "abs_FTIR"])

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"DNIR_v_FTIR_highlighted_plot1"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot6(save=False):
    """
    Compare discrete and continous ABS / PLA spectra. 
    """

    # Read data
    df1 = pd.read_csv("./data/plastic_2_hdpe_white.csv")
    df2 = pd.read_csv("./data/plastic_5_pp_white.csv")
    df3 = pd.read_csv("./data/plastic_6_ps_white.csv")
    df4 = pd.read_csv("./data/plastic_7_pla_white.csv")
    df5 = pd.read_csv("./data/empty.csv")

    dfs = [df1, df2, df3, df4, df5]
    mins = []
    maxs = []
    for df in dfs:
        
        # Drop extra column
        df = df.drop(['Unnamed: 0'], axis=1)
        # Map any none-zero values to zero
        df[df < 0] = 0.0

        mins.append(df.to_numpy().min())
        maxs.append(df.to_numpy().max())

        # print(df.head())

    min1 = min(mins)
    max1 = max(maxs)

    means = []
    for df in dfs:
        
        # Drop extra column
        df = df.drop(['Unnamed: 0'], axis=1)
        # Map any none-zero values to zero
        df[df < 0] = 0.0

        mean = np.asarray(( df.mean(axis=0).values - min1) / (max1 - min1))
        means.append(mean)

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    ax1.set_title(f"Comparing Plastic Types")

    labels = ['Type2 (HDPE)', 'Type5 (PP)', 'Type6 (PS)', 'Type7 (PLA)', 'Empty']
    colors = ['r', 'b', 'g', 'k', 'm']
    for i, m in enumerate(means):
        ax1.plot(xs, m, c=colors[i], label=labels[i])

    # Higher Intensity Plots
    indeces1 = [0, 1, 2, 7]
    for i, m in enumerate(means):
        xs1 = [xs[j] for j in indeces1]
        m1 = [m[j] for j in indeces1]
        ax2.scatter(xs1, m1, c=colors[i], label=labels[i])

    # Lower Intensity Plots
    indeces2 = [3, 4, 5, 6]
    for i, m in enumerate(means):
        xs2 = [xs[j] for j in indeces2]
        m2 = [m[j] for j in indeces2]
        ax3.scatter(xs2, m2, c=colors[i], label=labels[i])

    ax1.set_ylabel("Relative Intensity")
    ax3.set_xlabel("Wavelength (nm)")
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"plastics_white_1"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot7(save=False):
    """
    Compare different plastic types
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]

    Spectra = SpectraGen(led_wavelengths=leds)

    # Read data
    df_ref = pd.read_csv("./data/bv1_reference_al_block.csv")
    df_cali = pd.read_csv("./data/bv1_open_calibration.csv")

    # ref = Spectra.subtract_noise(df=df_ref)
    # cali = Spectra.subtract_noise(df=df_cali)

    df1 = pd.read_csv("./data/bv1_plastic_type_2_white.csv")
    df2 = pd.read_csv("./data/bv1_plastic_type_5_white.csv")
    df3 = pd.read_csv("./data/bv1_plastic_type_6_white.csv")
    df4 = pd.read_csv("./data/bv1_pla_white.csv")
    df5 = pd.read_csv("./data/bv1_abs_white.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    ax1.set_title(f"Comparing Plastic Types")

    labels = ['Type2 (HDPE)', 'Type5 (PP)', 'Type6 (PS)', 'Type7 (PLA)', 'Type7 (ABS)', 'Reference', 'Calibration']
    colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

    dfs = [df1, df2, df3, df4, df5, df_ref, df_cali]
    for i, df in enumerate(dfs):
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        ys = Spectra.get_values()
        
        # zip the two lists together
        zipped = list(zip(leds, ys))

        # sort the list of tuples based on the values in the first list
        sorted_zipped = sorted(zipped, key=lambda x: x[0])

        # unzip the sorted list of tuples back into separate lists
        xs, ys = zip(*sorted_zipped)

        ax1.plot(xs, ys, c=colors[i], label=labels[i])

    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_plastics_white"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot8(save=False):
    """
    Compare different plastic types
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]

    Spectra = SpectraGen(led_wavelengths=leds)

    # Read data
    df_ref = pd.read_csv("./data/bv1_plastic_type_2_white.csv")
    df_cali = pd.read_csv("./data/bv1_open_calibration.csv")
    df_ref.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)

    ref = Spectra.subtract_noise(df=df_ref)
    cali = Spectra.subtract_noise(df=df_cali)
    Spectra.add_calibration_values(cali)
    Spectra.add_reference_values(ref)

    df1 = pd.read_csv("./data/bv1_pla_white.csv")
    df2 = pd.read_csv("./data/bv1_abs_white.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    leds_ordered = sorted(leds)
    ax1.set_title(f"PLA vs ABS White, Leds: {leds_ordered}")

    labels = ['pla', 'abs']
    colors = ['c', 'm']

    dfs = [df1, df2]
    for i, df in enumerate(dfs):
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        Spectra.filtered_spectra()
        ys = Spectra.normalize()

        # zip & sort for nice plot
        zipped = list(zip(leds, ys))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        xs, ys = zip(*sorted_zipped)

        ax1.set_ylim(0, 1) # Relative intensity is bound [0,1]
        ax1.plot(xs, ys, c=colors[i], label=labels[i], marker='x', ms=7)

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_pla_v_abs_white"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot9(save=False):
    """
    Compare different plastic types
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]

    Spectra = SpectraGen(led_wavelengths=leds)

    # Read data
    df_ref = pd.read_csv("./data/bv1_plastic_type_2_white.csv")
    df_cali = pd.read_csv("./data/bv1_open_calibration.csv")
    df_ref.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)

    ref = Spectra.subtract_noise(df=df_ref)
    cali = Spectra.subtract_noise(df=df_cali)
    Spectra.add_calibration_values(cali)
    Spectra.add_reference_values(ref)

    df1 = pd.read_csv("./data/bv1_pla_white.csv")
    df2 = pd.read_csv("./data/bv1_abs_white.csv")
    df3 = pd.read_csv("./data/bv1_pla_black.csv")
    df4 = pd.read_csv("./data/bv1_abs_black.csv")
    
    # lime green is approx middle of spectrum
    df5 = pd.read_csv("./data/bv1_pla_lime.csv")
    df6 = pd.read_csv("./data/bv1_abs_lime.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    leds_ordered = sorted(leds)
    ax1.set_title(f"PLA vs ABS Colour Range, Leds: {leds_ordered}")

    labels = ['pla_white', 'abs_white', 'pla_black', 'abs_black', 'pla_lime', 'abs_lime']
    colors = ['k', 'r', 'k', 'r', 'k', 'r']
    linestyles = ['--', '-.', '--', '-.', '-', ':']
    markers = ['x', 'x', '>', '>', 'o', 'o']

    dfs = [df1, df2, df3, df4, df5, df6]
    for i, df in enumerate(dfs):
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        ys = Spectra.filtered_spectra()
        # ys = Spectra.normalize()

        # zip & sort for nice plot
        zipped = list(zip(leds, ys))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        xs, ys = zip(*sorted_zipped)

        ax1.set_ylim(0, 0.6) # Relative intensity is bound [0,1]
        ax1.plot(xs, ys, c=colors[i], label=labels[i], ls=linestyles[i], marker=markers[i], ms=7)

    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_pla_v_abs_colour_ranges"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot10(save=False):
    """
    Compare different plastic types
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]

    Spectra = SpectraGen(led_wavelengths=leds)

    # Read data
    df_ref = pd.read_csv("./data/bv1_reference_compiled.csv")
    df_cali = pd.read_csv("./data/bv1_open_calibration.csv")
    df_ref.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)

    ref = Spectra.subtract_noise(df=df_ref)
    cali = Spectra.subtract_noise(df=df_cali)
    Spectra.add_calibration_values(cali)
    Spectra.add_reference_values(ref)

    df1 = pd.read_csv("./data/bv1_pla_white.csv")
    df2 = pd.read_csv("./data/bv1_abs_white.csv")
    df3 = pd.read_csv("./data/bv1_pla_black.csv")
    df4 = pd.read_csv("./data/bv1_abs_black.csv")
    
    # lime green is approx middle of spectrum
    df5 = pd.read_csv("./data/bv1_pla_lime.csv")
    df6 = pd.read_csv("./data/bv1_abs_lime.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    leds_ordered = sorted(leds)
    ax1.set_title(f"PLA vs ABS Colour Range, Normalized, Leds: {leds_ordered}")

    labels = ['pla_white', 'abs_white', 'pla_black', 'abs_black', 'pla_lime', 'abs_lime']
    colors = ['k', 'r', 'k', 'r', 'k', 'r']
    linestyles = ['--', '-.', '--', '-.', '-', ':']
    markers = ['x', 'x', '>', '>', 'o', 'o']

    dfs = [df1, df2, df3, df4, df5, df6]
    for i, df in enumerate(dfs):
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        Spectra.filtered_spectra()
        ys = Spectra.normalize()

        # zip & sort for nice plot
        zipped = list(zip(leds, ys))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        xs, ys = zip(*sorted_zipped)

        ax1.set_ylim(0, 1) # Relative intensity is bound [0,1]
        ax1.plot(xs, ys, c=colors[i], label=labels[i], ls=linestyles[i], marker=markers[i], ms=7)

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_pla_v_abs_colour_ranges_normalized"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot11(save=False):
    """
    Compare different plastic types
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]
    Spectra = SpectraGen(led_wavelengths=leds)

    # calibration data
    calibration_files = {
        "1": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id1_daytime_calibration_2023_03_05.csv",
        "2": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id2_late_afternoon_calibration_2023_03_05.csv",
        "3": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id3_daytime_calibration_2023_03_06.csv",
    }
    df2 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/abs/bv1_id1_abs_white_2023_03_05_1678039435.csv")
    df4 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/pla/bv1_id1_pla_white_2023_03_05_1678042679.csv")
    df5 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/non_plastics/bv1_id3_aluminum_2023_03_06_1678121627.csv")
    df6 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/non_plastics/bv1_id3_clear_glass_2023_03_06_1678121914.csv")
    df7 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/non_plastics/bv1_id3_cold_rolled_steel_2023_03_06_1678121479.csv")
    df8 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/non_plastics/bv1_id3_hdf_board_2023_03_06_1678122181.csv")
    df9 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/petg/bv1_id1_petg_white_2023_03_05_1678047349.csv")
    df10 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/petg/bv1_id1_petg_red_2023_03_05_1678047709.csv")
    df11 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_2_white_2023_03_05_1678056134.csv")
    df12 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_4_white_2023_03_05_1678055943.csv")
    df14 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_5_white_2023_03_05_1678055866.csv")
    df15 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_6_white_2023_03_05_1678055532.csv")
    df16 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_unknown_pink_2023_03_05_1678056855.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    leds_ordered = sorted(leds)
    ax1.set_title(f"PLA, ABS, PETG, Plastics, Non_Plastics\nLeds: {leds_ordered}")

    labels = ['abs_white','pla_white', 'non_plastic_al', "non_plastic_glass", "non_plastic_steel", "non_plastic_hdf_board", "petg_white", "petg_red", "plastic_type_2_white", "plastic_type_4_white", "plastic_type_5_white", "plastic_type_6_white", "plastic_unknown"]
    colors = ['k', 'r', 'c', 'c', 'c', 'c', 'g', 'g', 'm', 'm', 'm', 'm', 'm']
    markers = ['o', 'v', '1', 's', 'p', '*', 'x', 'd', '|', '.', '+', 'h', 'P', 'D', 'X', '>']

    dfs = [df2, df4, df5, df6, df7, df8, df9, df10, df11, df12, df14, df15, df16]
    cali_ids = [1, 1, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2]

    for i, df in enumerate(dfs):

        # configure calibration
        df_cali = pd.read_csv(calibration_files[str(cali_ids[i])])
        df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        cali = Spectra.subtract_noise(df=df_cali)
        Spectra.add_calibration_values(cali)
        
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        ys = Spectra.filtered_spectra()

        # zip & sort for nice plot
        zipped = list(zip(leds, ys))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        xs, ys = zip(*sorted_zipped)

        ax1.set_ylim(0, 0.8) # Relative intensity is bound [0,1]
        ax1.plot(xs, ys, c=colors[i], label=labels[i], marker=markers[i], ms=7)

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_pla_v_abs_v_petg_v_plastics_v_non_plastics"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot12(save=False):
    """
    Compare different plastic types
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]
    Spectra = SpectraGen(led_wavelengths=leds)

    # calibration data
    calibration_files = {
        "1": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id1_daytime_calibration_2023_03_05.csv",
        "2": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id2_late_afternoon_calibration_2023_03_05.csv",
        "3": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id3_daytime_calibration_2023_03_06.csv",
    }

    df1 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/abs/bv1_id1_abs_black_2023_03_05_1678041025.csv")
    df3 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/pla/bv1_id1_pla_black_2023_03_05_1678041849.csv")
    df6 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/non_plastics/bv1_id3_clear_glass_2023_03_06_1678121914.csv")
    df7 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/non_plastics/bv1_id3_cold_rolled_steel_2023_03_06_1678121479.csv")
    df13 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_5_black_2023_03_05_1678056714.csv")
    df2 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_1_clear_2023_03_05_1678056482.csv")
    df4 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/plastics/bv1_id2_plastic_type_2_clear_2023_03_05_1678055181.csv")
    df5 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/val/other/plastics/bv1_id2_plastic_type_5_clear_2023_03_05_1678057773.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    leds_ordered = sorted(leds)
    ax1.set_title(f"PLA, ABS, Plastics, Non_Plastics (Black)\nLeds: {leds_ordered}")

    labels = ['abs_black', 'pla_black', "non_plastic_glass", "non_plastic_steel", "plastic_type_5_black", "plastic_type_1_clear", "plastic_type_2_clear", "plastic_type_5_clear"]
    colors = ['k', 'r', 'c', 'g', 'm', 'b', 'y', 'orange']
    markers = ['o', 'v', '1', 's', 'p', '*', 'x', 'd']

    dfs = [df1, df3, df6, df7, df13, df2, df4, df5]
    cali_ids = [1, 1, 3, 3, 2, 2, 2, 2]

    for i, df in enumerate(dfs):

        # configure calibration
        df_cali = pd.read_csv(calibration_files[str(cali_ids[i])])
        df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        cali = Spectra.subtract_noise(df=df_cali)
        Spectra.add_calibration_values(cali)
        
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        ys = Spectra.filtered_spectra()

        # zip & sort for nice plot
        zipped = list(zip(leds, ys))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        xs, ys = zip(*sorted_zipped)

        ax1.set_ylim(0, 0.2) # Relative intensity is bound [0,1]
        ax1.plot(xs, ys, c=colors[i], label=labels[i], marker=markers[i], ms=7)

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_pla_v_abs_v_plastics_v_non_plastics_black"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def plot13(save=False):
    """
    Compare abs, pla, petg
    """

    # led config
    leds = [850, 940, 1050, 890, 1300, 880, 1550, 1650]
    Spectra = SpectraGen(led_wavelengths=leds)

    # calibration data
    calibration_files = {
        "1": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id1_daytime_calibration_2023_03_05.csv",
        "2": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id2_late_afternoon_calibration_2023_03_05.csv",
        "3": "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id3_daytime_calibration_2023_03_06.csv",
    }
    df1 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/abs/bv1_id1_abs_white_2023_03_05_1678039435.csv")
    df2 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/pla/bv1_id1_pla_white_2023_03_05_1678042679.csv")
    df3 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/abs/bv1_id1_abs_black_2023_03_05_1678041025.csv")
    df4 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/pla/bv1_id1_pla_black_2023_03_05_1678041849.csv")
    df5 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/petg/bv1_id1_petg_white_2023_03_05_1678047349.csv")
    df6 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/other/petg/bv1_id1_petg_red_2023_03_05_1678047709.csv")
    df7 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/val/other/petg/bv1_id1_petg_orange_2023_03_05_1678048368.csv")
    df8 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/abs/bv1_id1_abs_red_2023_03_05_1678040622.csv")
    df9 = pd.read_csv("/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/train/pla/bv1_id1_pla_red_2023_03_05_1678042799.csv")

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=True)

    # Overall Plot
    leds_ordered = sorted(leds)
    ax1.set_title(f"PLA, ABS, PETG\nLeds: {leds_ordered}")

    labels = ['abs_white','pla_white', 'abs_black', 'pla_black', 'petg_white', 'petg_red', 'petg_orange', 'abs_red', 'pla_red']
    colors = ['k', 'r', 'k', 'r', 'b', 'b', 'b', 'k', 'r']
    markers = ['o', 'v', '1', 's', 'p', '*', 'x', 'd', '>']

    dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
    cali_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    for i, df in enumerate(dfs):

        # configure calibration
        df_cali = pd.read_csv(calibration_files[str(cali_ids[i])])
        df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        cali = Spectra.subtract_noise(df=df_cali)
        Spectra.add_calibration_values(cali)
        
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        Spectra.add_measurements(df)
        ys = Spectra.filtered_spectra()

        # zip & sort for nice plot
        zipped = list(zip(leds, ys))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        xs, ys = zip(*sorted_zipped)

        ax1.set_ylim(0, 0.8) # Relative intensity is bound [0,1]
        ax1.plot(xs, ys, c=colors[i], label=labels[i], marker=markers[i], ms=7)

    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"bv1_pla_v_abs_v_petg"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()
    
    
def main():
    print("Generating Plots...")

    # plot1(save=True)
    # plot2(save=True)
    # plot3(save=True)
    # plot4(save=True)
    # plot5(save=True)
    # plot6(save=True)

    # Using new data format and boards
    # plot7(save=True)
    # plot8(save=True)
    # plot9(save=True)
    # plot10(save=True)
    # plot11(save=True)
    # plot12(save=True)
    plot13(save=True)

if __name__ == "__main__":
    main()
