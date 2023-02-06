""" Working script to analyze Discrete NIR Data """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def main():
    print("Generating Plots...")

    # plot1(save=True)
    # plot2(save=True)
    # plot3(save=True)
    # plot4(save=True)
    # plot5(save=True)
    plot6(save=True)

if __name__ == "__main__":
    main()