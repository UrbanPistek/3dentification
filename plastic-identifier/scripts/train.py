import os
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
DATA_DIR = "./data"
TRAIN_DIR = DATA_DIR + "/train"
VAL_DIR = DATA_DIR + "/val"

def init_spectra_cal_ref(S: SpectraGen) -> None:

    # Read data
    df_ref = pd.read_csv("./data/bv1_reference_compiled.csv")
    df_cali = pd.read_csv("./data/bv1_open_calibration.csv")
    df_ref.rename(columns={"Unnamed: 0": "units"}, inplace=True)
    df_cali.rename(columns={"Unnamed: 0": "units"}, inplace=True)

    ref = S.subtract_noise(df=df_ref)
    cali = S.subtract_noise(df=df_cali)
    S.add_calibration_values(cali)
    S.add_reference_values(ref)

def gen_datasets(Spectra: SpectraGen) -> np.ndarray:
    
    # training dataset
    train_x, train_y = [], []
    train_files_pla = glob.glob(TRAIN_DIR + "/pla/**/*.csv")
    train_files_abs = glob.glob(TRAIN_DIR + "/abs/**/*.csv")
    
    test_x, test_y = [], []
    test_files_pla = glob.glob(VAL_DIR + "/pla/*.csv")
    test_files_abs = glob.glob(VAL_DIR + "/abs/*.csv")

    # add pla
    # pla = 0
    # abs = 1
    extract_data(train_x, train_y, Spectra, 0, train_files_pla)
    extract_data(test_x, test_y, Spectra, 0, test_files_pla)

    # add abs
    extract_data(train_x, train_y, Spectra, 1, train_files_abs)
    extract_data(test_x, test_y, Spectra, 1, test_files_abs)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def extract_data(x: list, y: list, S: SpectraGen, label: int, glob_files): 

    for file in glob_files:
        df = pd.read_csv(file)
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        S.add_measurements(df)
        S.filtered_spectra()
        ys = S.normalize()

        x.append(ys)
        y.append(label)

def main():
    print("Spectra Classifier Training...")
    
    # Spectra generation object
    Spectra = SpectraGen(led_wavelengths=LEDS)

    # Initialize spectra object
    init_spectra_cal_ref(Spectra)
    Spectra.display()

    train_x, train_y, test_x, test_y = gen_datasets(Spectra)
    print(f"\ntrain_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")

    names = [
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    ]

    classifiers = [
        DecisionTreeClassifier(max_depth=25),
        RandomForestClassifier(max_depth=15, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
    ]

    for name, clf in zip(names, classifiers):

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_x, train_y)
        score = clf.score(test_x, test_y)

        print(f"CLF: {name}\n Score: {score}")

if __name__ == "__main__":
    main()
