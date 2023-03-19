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

# Models
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV

# Configure prints
np.set_printoptions(precision=7, suppress=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id3_daytime_calibration_2023_03_06.csv",
    "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/bv1_id4_late_afternoon_calibration_2023_03_15.csv",
]

# Load colour data
COLOUR_DATA = "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/data/dataset1/colours.json"
with open(COLOUR_DATA, 'r') as f:
    data = f.read()
    COLOUR_DICT = json.loads(data)

# Directories for each category
# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/non_plastics", "/other/petg", "/other/plastics"] 
# LABELS = [0, 1, 2, 3, 4, 5]
# LABEL_NAMES = ["abs", "pla", "empty", "non_plastics", "petg", "plastics"]

# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/non_plastics", "/other/plastics"] 
# LABELS = [0, 1, 2, 3, 4]
# LABEL_NAMES = ["abs", "pla", "empty", "non_plastics", "plastics"]

# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/petg", "/other/plastics"] 
# LABELS = [0, 1, 2, 3, 4]
# LABEL_NAMES = ["abs", "pla", "empty", "petg", "plastics"]

# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/petg", "/other/non_plastics"] 
# LABELS = [0, 1, 2, 3, 4]
# LABEL_NAMES = ["abs", "pla", "empty", "petg", "non_plastics"]

# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/plastics"] 
# LABELS = [0, 1, 2, 3]
# LABEL_NAMES = ["abs", "pla", "empty", "plastics"]

# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/non_plastics"] 
# LABELS = [0, 1, 2, 3]
# LABEL_NAMES = ["abs", "pla", "empty", "non_plastics"]

LABEL_DIRS = ["/abs", "/pla", "/empty", "/other/petg"] 
LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["abs", "pla", "empty", "petg"]

# LABEL_DIRS = ["/abs", "/pla", "/empty", "/other"]
# LABELS = [0, 1, 2, 3]
# LABEL_NAMES = ["abs", "pla", "empty", "other"]

# LABEL_DIRS = ["/abs", "/pla", "/empty"]
# LABELS = [0, 1, 2]
# LABEL_NAMES = ["abs", "pla", "empty"]

# LABEL_DIRS = ["/abs", "/pla", "/other"]
# LABELS = [0, 1, 2]
# LABEL_NAMES = ["abs", "pla", "other"]

# LABEL_DIRS = ["/abs", "/pla"]
# LABELS = [0, 1]
# LABEL_NAMES = ["abs", "pla"]

# For regex
CALI_ID_REGEX =  r"_id(\d+)_"

# For colour data 
COLOUR_REGEX = r"_id[0-9]_(.*?)_2023"

def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser.parse_args:
    parser.add_argument(
        "-v",
        "--verbose", 
        action='store_true',
        help='Run training in verbose, showing confusion matrices and outputs')
    parser.add_argument(
        "-s",
        "--save", 
        action='store_true',
        help='Save model')
    parser.add_argument(
        "-o",
        "--optimize", 
        action='store_true',
        help='Run optimization on a set of models')
    parser.add_argument(
        "-t",
        "--train", 
        action='store_true',
        help='Train a set of models')

    return parser.parse_args()

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
        else:
            material = match.split('_', 1)[0]

        dir_type = "train"
        if "val" in file:
            dir_type = "val"
        
        idx = COLOUR_DICT[material][dir_type]["scan"].index(match)
        rgb = COLOUR_DICT[material][dir_type]["colour"][idx]

        return np.asarray(rgb, dtype=int)
    except:
        print(f"==> ERROR: {file}")

def extract_data(x: list, y: list, S: SpectraGen, label: int, glob_files, enable_ratios_vec=False, enable_colour_data=False) -> None: 

    for file in glob_files:
        df = pd.read_csv(file)
        df.rename(columns={"Unnamed: 0": "units"}, inplace=True)
        S.add_measurements(df)
        ys = S.filtered_spectra()

        # Add ratios vector
        if enable_ratios_vec:
            ratios_vec = S.create_ratios_vector()

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

            ys = np.concatenate((ys, ratios_vec), axis=0)

        # Get colour data
        if enable_colour_data:
            rgb = add_colour_data(file)
            rgb = (rgb) / (255) # MinMax scaling
            
            ys = np.concatenate((ys, rgb), axis=0)

        x.append(ys)
        y.append(label)

def merge_data(Spectra: SpectraGen, directory: str, label: int, calibrationId: int, x: list, y: list) -> None:

    # Grab all files in the directory according to a specific calibration id
    files = glob.glob(directory + f"/**/*_id{calibrationId}_*.csv", recursive=True)
    extract_data(x, y, Spectra, label, files, enable_ratios_vec=True, enable_colour_data=True)

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
    print("Spectra Classifier Training...")
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    
    # Spectra generation object
    Spectra = SpectraGen(led_wavelengths=LEDS)

    train_x, train_y, test_x, test_y = gen_datasets(Spectra)
    print(f"\ntrain_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")
    print(f"Inputs: ", train_x[0])
    print(f"Outputs: ", train_y[0])

    # ===========================[ Training ]==============================
    if args.train: 
        print(f"Training for: {LABEL_NAMES}")

        names = [
            "Decision Tree",
            "Random Forest",
            "Bagging Classifier",
            "Extra Trees Classifier",
            "Gradient Boosting Classifier",
            "Voting Classifier",
            "Histogram Gradient Boosting Classifier",
            "AdaBoost",
            "K Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "MLP Classifier",
            "QDA",
        ]

        classifiers = [
            DecisionTreeClassifier(max_depth=25),
            RandomForestClassifier(max_depth=25, n_estimators=25, max_features=8),
            BaggingClassifier(estimator=SVC(),n_estimators=10, random_state=0),
            ExtraTreesClassifier(n_estimators=25, random_state=0),
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0, max_depth=1, random_state=0),
            VotingClassifier(estimators=[
                ('dt', DecisionTreeClassifier(max_depth=25)),
                ('rf', RandomForestClassifier(max_depth=25, n_estimators=25, max_features=8)), 
                ('bag', BaggingClassifier(estimator=SVC(),n_estimators=10, random_state=0)),
                ('knn', KNeighborsClassifier(len(LABEL_DIRS))), 
                ('mlp', MLPClassifier(alpha=1, max_iter=10000)),
                ('ada', AdaBoostClassifier()),
                ('hgb', HistGradientBoostingClassifier()),
                ], voting='hard'),
            HistGradientBoostingClassifier(),
            AdaBoostClassifier(),
            KNeighborsClassifier(n_neighbors=len(LABEL_DIRS)),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            MLPClassifier(alpha=1, max_iter=10000),
            QuadraticDiscriminantAnalysis(),
        ]

        for name, clf in zip(names, classifiers):

            # clf = make_pipeline(StandardScaler(), clf)
            # clf = make_pipeline(MinMaxScaler(), clf)
            clf.fit(train_x, train_y)
            score = clf.score(test_x, test_y)

            # save model
            if args.save:
                if not os.path.exists('temp'):
                    os.makedirs('temp')

                m_name = name.replace(" ", "")
                score_str = str(round(score, 2) - int(round(score, 3)))[1:].replace(".", "")
                cat_labels = "_".join(LABEL_NAMES)
                filename = f"model_{m_name}_{score_str}_{cat_labels}"
                with open(f'./temp/{filename}.pickle', 'wb') as file:
                    # Pickle the object and save it to the file
                    pickle.dump(clf, file)

            # Display confusion Matrix
            if args.verbose:
                fig, ax = plt.subplots(figsize=(12,7))
                ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y, display_labels=LABEL_NAMES)
                
                ax.set_title(f"{name}: Confusion Matrix")
                # save plot
                if not os.path.exists('figures'):
                    os.makedirs('figures')
                m_name = name.replace(" ", "")
                filename = f"cfm_{m_name}"
                plt.savefig(f'figures/{filename}.png')

            print(f"CLF: {name}\n Score: {score}")

    # ===========================[ Optimization ]==============================
    elif args.optimize:

        names = [
            "KNN",
        ]

        classifiers = [
            KNeighborsClassifier(len(LABEL_DIRS)),
        ]

        param_dist = [
            {
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": list(range(4, 80, 2)),
                "p": [1, 2],
            },
        ]

        print(f"Optimizing for: {names}")
        
        # Combine train and test
        X = np.concatenate((train_x, test_x), axis=0)
        y = np.concatenate((train_y, test_y), axis=0)
        print(f"\nX: {X.shape}, y: {y.shape}")

        for i, clf in enumerate(classifiers):

            rsh = HalvingRandomSearchCV(
                estimator=clf, param_distributions=param_dist[i], factor=2, verbose=3
            )
            rsh.fit(X, y)

            # Plot results of search
            results = pd.DataFrame(rsh.cv_results_)
            results["params_str"] = results.params.apply(str)
            results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
            mean_scores = results.pivot(
                index="iter", columns="params_str", values="mean_test_score"
            )
            ax = mean_scores.plot(legend=False, alpha=0.8, figsize=(16,9))

            labels = [
                f"iter={i}\n" for i in range(rsh.n_iterations_)
            ]

            ax.set_xticks(range(rsh.n_iterations_))
            ax.set_xticklabels(labels, multialignment="left")
            ax.set_title(f"{names[i]}\nBest: {rsh.best_params_}\nScore: {rsh.best_score_}")
            ax.set_ylabel("mean test score", fontsize=15)
            ax.set_xlabel("iterations", fontsize=15)

            # save plot
            if not os.path.exists('figures'):
                os.makedirs('figures')
            m_name = names[i].replace(" ", "")
            filename = f"optimize_{m_name}_random_halving_search"
            plt.savefig(f'figures/{filename}.png')

            print(f"\n=> {names[i]}\nBest: {rsh.best_params_}\nScore: {rsh.best_score_}\n")

    else:
        parser.print_usage()
        parser.print_help()

if __name__ == "__main__":
    main()
