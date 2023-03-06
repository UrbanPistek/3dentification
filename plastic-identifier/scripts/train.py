import os
import re
import sys
import time
import json
import glob
import pandas as pd
import numpy as np

# Models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Configure prints
np.set_printoptions(precision=7, suppress=True)

# Internal modules
from lib.postprocess import SpectraGen

# Leds used
LEDS = [850, 940, 1050, 890, 1300, 880, 1550, 1650]

# Locations of data
DATA_DIR = "./data/dataset1"
TRAIN_DIR = DATA_DIR + "/train"
VAL_DIR = DATA_DIR + "/val"
CALIBRATION_FILES = [
    "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id1_daytime_calibration_2023_03_05.csv",
    "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id2_late_afternoon_calibration_2023_03_05.csv",
]

# Directories for each category
# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other"]
# LABELS = [0, 1, 2, 3]
LABEL_DIRS = ["/abs", "/pla"]
LABELS = [0, 1]

# For regex
CALI_ID_REGEX =  r"_id(\d+)_"

def init_spectra_cal_ref(S: SpectraGen, calibration_file: str) -> None:

    # Read data
    df_cali = pd.read_csv(calibration_file)

    # Remove noise and add calibration
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    cali = S.subtract_noise(df=df_cali)
    S.add_calibration_values(cali)

def extract_data(x: list, y: list, S: SpectraGen, label: int, glob_files): 

    for file in glob_files:
        df = pd.read_csv(file)
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        S.add_measurements(df)
        ys = S.filtered_spectra()

        x.append(ys)
        y.append(label)

def merge_data(Spectra: SpectraGen, directory: str, label: int, calibrationId: int, x: list, y: list):

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

def main():
    print("Spectra Classifier Training...")
    
    # Spectra generation object
    Spectra = SpectraGen(led_wavelengths=LEDS)

    train_x, train_y, test_x, test_y = gen_datasets(Spectra)
    print(f"\ntrain_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")

    names = [
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    ]

    classifiers = [
        DecisionTreeClassifier(max_depth=250),
        RandomForestClassifier(max_depth=250, n_estimators=50, max_features=8),
        AdaBoostClassifier(),
    ]

    for name, clf in zip(names, classifiers):

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_x, train_y)
        score = clf.score(test_x, test_y)

        print(f"CLF: {name}\n Score: {score}")

if __name__ == "__main__":
    main()
