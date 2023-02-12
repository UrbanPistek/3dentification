import os
import sys
import serial
import time
import json
import pandas as pd

from lib.utils import get_serial_ports
from utils.read_serial import write_read

# Specify file to save data to
FILENAME = 'sample_test.csv'

def main():

    # Print out all availible ports
    ports = get_serial_ports()
    print(f"Availible Ports: \n{ports}\n")
    print(f"Reading from port: \n{ports[0]}\n")

    # Using long timeout to wait for scan data
    arduino = serial.Serial(port=ports[0], baudrate=115200, timeout=5)

    ts = time.time()

    value = write_read(arduino, "ping")
    print(f"Ping:\n{value}")
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

    if not os.path.exists('data'):
        os.makedirs('data')

    df = pd.DataFrame.from_dict(data_dict)
    df.index = ['intensity', 'variance', 'ambient']
    df.to_csv(f"./data/{FILENAME}")
    print(df.head())
    
    te = time.time()
    print(f"\nElapsed time: {te - ts}s")

if __name__ == "__main__":
    main()

