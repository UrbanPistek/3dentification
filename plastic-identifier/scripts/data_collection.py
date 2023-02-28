import os
import sys
import serial
import time
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from lib.utils import get_serial_ports
from utils.read_serial import write_read

# Configure prints
np.set_printoptions(precision=9, suppress=True)

# Specify file to save data to
FILENAME = 'bv1_pla_dark_pink'
DATA_DIR = './data/train/pla/batch'
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

def get_data(verbose=True) -> None:

    # Print out all availible ports
    ports = get_serial_ports()
    verbose and print(f"Availible Ports: \n{ports}\n")
    verbose and print(f"Reading from port: \n{ports[0]}\n")

    # Using long timeout to wait for scan data
    arduino = serial.Serial(port=ports[0], baudrate=115200, timeout=5)

    ts = time.time()

    value = write_read(arduino, "ping")
    verbose and print(f"Ping:\n{value}")
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

    try:
        data = write_read(arduino, "gen_spectra")
        # print(f"Gen Spectra:\n{data}")   

        if sys.getsizeof(data) > 33: # check against empty

            # Decode byte object
            data_decoded = data.decode('utf-8')

            # convert to dict
            readings = json.loads(data_decoded)

            for key in data_dict:

                data_dict[key].append(readings[key])
                data_dict[key].append(readings[f"{key}_var"])
                data_dict[key].append(readings[f"{key}_ambient"])

    except:
        print("ERROR: read")
        return 

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    df = pd.DataFrame.from_dict(data_dict)
    df.index = ['intensity', 'variance', 'ambient']
    df.to_csv(f"{DATA_DIR}/{FILENAME}_{round(ts)}.csv")
    verbose and print(df.head())
    
    te = time.time()
    verbose and print(f"\nElapsed time: {te - ts}s")
    verbose and print(f"data saved to: {DATA_DIR}/{FILENAME}_{round(ts)}.csv")

def main():
    
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    print(f"=> collecting data for: {FILENAME}")

    if args.single:
        get_data()
    elif args.batch:
        for _ in tqdm(range(BATCH_SIZE)):
            get_data(verbose=False)
    else:
        parser.print_usage()
        parser.print_help()

if __name__ == "__main__":
    main()

