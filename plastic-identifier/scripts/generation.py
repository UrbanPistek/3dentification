import os
import re
import sys
import time
import json
import glob
import pickle
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure prints
np.set_printoptions(precision=7, suppress=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Internal modules
from lib.postprocess import SpectraGen

# Leds used
LEDS = [850, 940, 1050, 890, 1300, 880, 1550, 1650]

SAMPLE_DATA_SIZE = 10

# Locations of data
DATA_DIR = "./data/dataset3"
TRAIN_DIR = DATA_DIR + "/train"
VAL_DIR = DATA_DIR + "/val"
CALIBRATION_FILES = [
    "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset3/bv1_id1_afternoon_calibration_2023_03_21_1679427508.csv",
]

# Load colour data
COLOUR_DATA = "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset3/colours.json"
with open(COLOUR_DATA, 'r') as f:
    data = f.read()
    COLOUR_DICT = json.loads(data)

# Directories for each category
LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/non_plastics", "/other/petg", "/other/plastics"] 
LABELS = [0, 1, 2, 3, 4, 5]
LABEL_NAMES = ["abs", "pla", "empty", "non_plastics", "petg", "plastics"]

# For regex
CALI_ID_REGEX =  r"_id(\d+)_"

# For colour data 
COLOUR_REGEX = r"_id[0-9]_(.*?)_2023"

def init_spectra_cal_ref(S: SpectraGen, calibration_file: str) -> None:

    # Read data
    df_cali = pd.read_csv(calibration_file)

    # Remove noise and add calibration
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    cali = S.subtract_noise(df=df_cali)
    S.add_calibration_values(cali)

def add_colour_data(file: str) -> np.ndarray:
    """
    Bit janky, but quickly adding colour data after the fact.
    """

    try:
        # Extract info from file name
        match = re.search(COLOUR_REGEX, file).group(1)
    
        # Split to get material type
        if "empty" in match:
            material = "empty"
        elif "non_plastics" in file:
            material = "non_plastics"
        else:
            material = match.split('_', 1)[0]

            if material == "plastic":
                material = material + "s"

        dir_type = "train"
        if "val" in file:
            dir_type = "val"
        
        idx = COLOUR_DICT[material][dir_type]["scan"].index(match)
        rgb = COLOUR_DICT[material][dir_type]["colour"][idx]

        return np.asarray(rgb, dtype=int)
    except:
        print(f"==> ERROR: {file}")

def extract_data(x: list, y: list, S: SpectraGen, label: int, glob_files) -> None: 

    for file in glob_files:

        df = pd.read_csv(file)
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        S.add_measurements(df)
        values = S.filtered_spectra()
        noise = S.get_ambient_noise()
        var = S.get_variances()
        rgb = add_colour_data(file)

        ys = []
        xs = []

        for i in range(SAMPLE_DATA_SIZE):

            # Add some noise to ensure colours values are all not exactly the same
            sample_rgb = [x + random.randint(-15, 15) for x in rgb]
            
            # Replace values less than 0 with 0 and greater than 255 with 255
            sample_rgb = [0 if value < 0 else 255 if value > 255 else value for value in rgb]

            sample_values = [x + np.random.normal(loc=0, scale=np.sqrt(var[i])) for i,x in enumerate(values)]

            # Add ratios vector
            dim = len(sample_values)
            ratio_matrix = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if sample_values[j] == 0:
                        ratio_matrix[i,j] = 0
                    else:
                        ratio_matrix[i,j] = sample_values[i] / sample_values[j]

            # Flatten the matrix into a 1D array using the ravel() method
            ratios_vec = ratio_matrix.ravel()

            # Manually normalize the data
            max_ratio = max(ratios_vec)
            min_ratio = min(ratios_vec)

            for i in range(len(ratios_vec)):
                if (max_ratio - min_ratio) == 0:
                    ratios_vec[i] = 0
                elif ratios_vec[i] == 0:
                    ratios_vec[i] = 0
                else:
                    ratios_vec[i] = (ratios_vec[i] - min_ratio) / (max_ratio - min_ratio) # MinMax scaling

            arrs = [sample_values, ratios_vec, sample_rgb]
            vec = np.concatenate(arrs)

            ys.append(vec)
            xs.append(label)

        ys = np.asarray(ys)
        xs = np.asarray(xs)
        x.append(ys)
        y.append(label)

def merge_data(Spectra: SpectraGen, directory: str, label: int, calibrationId: int, x: list, y: list) -> None:

    # Grab all files in the directory according to a specific calibration id
    files = glob.glob(directory + f"/**/*_id{calibrationId}_*.csv", recursive=True)
    extract_data(x, y, Spectra, label, files)

def gen_datasets(Spectra: SpectraGen) -> np.ndarray:
    
    # training dataset
    train_x, train_y = [], []    
    test_x, test_y = [], []

    for calibration in CALIBRATION_FILES:

        # Get calibration id
        substring = re.search(CALI_ID_REGEX, calibration)
        cali_id = substring.group(1)

        # Initialize spectra object
        init_spectra_cal_ref(Spectra, calibration_file=calibration)

        # Merge data according to calibration ids
        # Run for training
        for i, dir in enumerate(LABEL_DIRS):
            merge_data(Spectra, TRAIN_DIR+dir, i, cali_id, train_x, train_y)

        # Run for validation
        for i, dir in enumerate(LABEL_DIRS):
            merge_data(Spectra, VAL_DIR+dir, i, cali_id, test_x, test_y)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def main() -> None:
    print("Spectra Data Generation...")
    
    # ===========================[ Pre-processing ]==============================
    # Spectra generation object
    Spectra = SpectraGen(led_wavelengths=LEDS)

    train_x, train_y, test_x, test_y = gen_datasets(Spectra)
    print(f"\ntrain_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")

if __name__ == "__main__":
    main()
