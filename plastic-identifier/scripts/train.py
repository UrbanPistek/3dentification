import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Configure prints
np.set_printoptions(precision=7, suppress=True)

# Internal modules
from lib.postprocess import SpectraGen, subtract_noise 

def main():
    print("Spectra Classifier Training...")

    reference = "./data/boardv1_white_abs.csv"
    measurements = "./data/boardv1_mock_abs.csv"
    calibrations = "./data/boardv1_open_calibration.csv"

    led_wavelengths = np.array([850, 940, 1050, 1200, 1300, 1460, 1550, 1650])

    df_vals = pd.read_csv(measurements)
    df_cali = pd.read_csv(calibrations)
    df_ref  = pd.read_csv(reference)

    # Rename unnamed column
    df_vals.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    df_ref.rename(columns={"Unnamed: 0": "units"}, inplace=True)

    print("\nMeasurements: \n", df_vals.head())
    print("\nCalibrations: \n", df_cali.head())
    print("\nReference: \n", df_ref.head())

    # ================== [Reference] ==================
    # convert to numpy array to perform calculations
    ref = df_ref.drop(['units'], axis=1)
    ref = ref.to_numpy()
    print(f"\nref shape: {ref.shape}")

    # Subtract out noise
    ref_signal = ref[0] - ref[2]
    ref_signal[ref_signal < 0] = 0 # Any negative values from the subtraction get mapped to zero
    print(f"ref intensity: {ref[0]} \nref noise: {ref[2]} \nref signal: {ref_signal}")
    
    # ================== [Calibration] ==================
    # convert to numpy array to perform calculations
    cali = df_cali.drop(['units'], axis=1)
    cali = cali.to_numpy()
    print(f"\ncali shape: {cali.shape}")

    # Subtract out noise
    cali_signal = cali[0] - cali[2]
    cali_signal[cali_signal < 0] = 0 # Any negative values from the subtraction get mapped to zero
    print(f"ref intensity: {cali[0]} \nref noise: {cali[2]} \nref signal: {cali_signal}")
    
    # ================== [Measurements] ==================
    vals_signal = subtract_noise(df_vals)
    print(f"\nvals signal: {vals_signal}")

if __name__ == "__main__":
    main()
