import os
import sys
import serial
import time
import json
import traceback
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from lib.utils import get_serial_ports
from utils.read_serial import write_read

# Configure prints
np.set_printoptions(precision=9, suppress=True)

# Specify file to save data to
CALIBRATION_ID = 0
BOARD_VERSION = 1
FILENAME = f'bv{BOARD_VERSION}_id{CALIBRATION_ID}_' + 'temp'
DATA_DIR = './data/dataset1/'
BATCH_SIZE = 10

def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser.parse_args:
    parser.add_argument(
        "-s",
        "--single", 
        action='store_true',
        help='Perform a single measurement')
    parser.add_argument(
        "-b",
        "--batch", 
        action='store_true',
        help='Perform multiple measurements')

    return parser.parse_args()

def get_data(verbose=True, readings=1, batch=False) -> None:

    # Print out all availible ports
    ports = get_serial_ports()
    verbose and print(f"Availible Ports: \n{ports}\n")
    verbose and print(f"Reading from port: \n{ports[0]}\n")

    # Using long timeout to wait for scan data
    arduino = serial.Serial(port=ports[0], baudrate=115200, timeout=5)

    ts = time.time()

    value = write_read(arduino, "ping")
    verbose and print(f"Ping:\n{value}")

    try:
        with ThreadPoolExecutor(max_workers=readings) as executor:
            
            for _ in tqdm(range(readings)):

                data = write_read(arduino, "gen_spectra")

                if sys.getsizeof(data) > 33: # check against empty

                    # Save data to file
                    executor.submit(save_data, data, verbose, batch)

    except Exception:
        print("ERROR: read")
        print(traceback.format_exc())
        return 
    
    te = time.time()
    print(f"\nElapsed time: {te - ts}s")

def get_scan() -> pd.DataFrame:

    # Print out all availible ports
    ports = get_serial_ports()

    # Using long timeout to wait for scan data
    arduino = serial.Serial(port=ports[0], baudrate=115200, timeout=5)
    
    # wake up serial
    write_read(arduino, "ping")

    try:
        data = write_read(arduino, "gen_spectra")
        if sys.getsizeof(data) > 33: # check against empty

            data_dict = {
                "led0": [],
                "led1": [],
                "led2": [],
                "led3": [],
                "led4": [],
                "led5": [],
                "led6": [],
                "led7": [],
            } 

            # Decode byte object
            data_decoded = data.decode('utf-8')

            # convert to dict
            readings = json.loads(data_decoded)

            for key in data_dict:

                data_dict[key].append(readings[key])
                data_dict[key].append(readings[f"{key}_var"])
                data_dict[key].append(readings[f"{key}_ambient"])

            data_dict["units"] = ['intensity', 'variance', 'ambient']
            df = pd.DataFrame.from_dict(data_dict)

            return df

    except Exception:
        print("ERROR: get scan")
        print(traceback.format_exc())
        return 

def save_data(data, verbose=False, batch=False):

    data_dict = {
        "led0": [],
        "led1": [],
        "led2": [],
        "led3": [],
        "led4": [],
        "led5": [],
        "led6": [],
        "led7": [],
    } 

    # Decode byte object
    data_decoded = data.decode('utf-8')

    # convert to dict
    readings = json.loads(data_decoded)

    for key in data_dict:

        data_dict[key].append(readings[key])
        data_dict[key].append(readings[f"{key}_var"])
        data_dict[key].append(readings[f"{key}_ambient"])

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    df = pd.DataFrame.from_dict(data_dict)
    df.index = ['intensity', 'variance', 'ambient']

    datestamp = datetime.now().strftime('%Y/%m/%d').replace('/', '_').replace(' ', '_')
    if batch:
        ts = time.time()
        datestamp = datestamp + '_' + str(round(ts))

    df.to_csv(f"{DATA_DIR}/{FILENAME}_{datestamp}.csv")
    verbose and print(df.head())
    verbose and print(f"data saved to: {DATA_DIR}/{FILENAME}_{datestamp}.csv")

def main():
    
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    print(f"=> collecting data in {DATA_DIR} , for: {FILENAME}")

    if args.single:
        get_data()
    elif args.batch:
        get_data(verbose=False, readings=BATCH_SIZE, batch=True)
    else:
        parser.print_usage()
        parser.print_help()

if __name__ == "__main__":
    main()

