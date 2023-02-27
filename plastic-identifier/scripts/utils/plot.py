""" Plot Discrete NIR Data """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(df: pd.DataFrame, save=False):

    # plot
    fig1, (ax1) = plt.subplots(1, 1, figsize=(12,7), sharex=True)

    # Caclulate trendlines
    xs = df['wavelength'].values

    # Means & Best plot
    ax1.set_title(f"Discrete NIR Spectra")
    ax1.plot(xs, df['baseline'].values, c='k', label='baseline')
    ax1.plot(xs, df['pla_blue'].values, c='c', label='pla_blue')
    ax1.plot(xs, df['pla_white'].values, c='g', label='pla_white')
    ax1.plot(xs, df['abs_red'].values, c='r', label='abs_red')
    ax1.plot(xs, df['petg_clear'].values, c='m', label='petg_clear')
    
    ax1.set_ylabel("Relative Intensity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend()

    # save plot and results
    if save:
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # save plot
        filename = f"nir_data_plot"
        plt.savefig(f'figures/{filename}.png')
    else:
        plt.show()

def main():
    print("Plotting...")

    # Read data
    df = pd.read_csv("./data/fydp_initial_NIR_data.csv")

    # Map any none-zero values to zero
    df[df < 0] = 0.0
    print(df.head(10))

    # only values
    # val_df = df['baseline', 'pla_blue', 'pla_white', 'abs_red', 'petg_clear', '']
    val_df = df.drop(['led', 'wavelength'], axis=1)
    print(val_df.head(10))

    # apply min/max scaling
    min = val_df.to_numpy().min()
    max = val_df.to_numpy().max()
    print(f"min: {min}, max: {max}")

    df["baseline"] = (df["baseline"] - min) / (max - min)
    df["pla_blue"] = (df["pla_blue"] - min) / (max - min)
    df["pla_white"] = (df["pla_white"] - min) / (max - min)
    df["abs_red"] = (df["abs_red"] - min) / (max - min)
    df["petg_clear"] = (df["petg_clear"] - min) / (max - min)
    print(df.head(10))

    # plot data
    plot(df, save=True)

if __name__ == "__main__":
    main()