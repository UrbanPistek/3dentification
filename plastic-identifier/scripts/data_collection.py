import os
import sys
import serial
import time
import json
import pandas as pd

from lib.utils import get_serial_ports
from read_serial import write_read, write_read_blocking

# Number of readings to perform
NUM_READINGS = 10
FILENAME = 'sample_test_run.csv'

def main():

    # Print out all availible ports
    ports = get_serial_ports()
    print(f"Availible Ports: \n{ports}\n")
    print(f"Reading from port: \n{ports[0]}\n")

    # Using long timeout to wait for scan data
    arduino = serial.Serial(port=ports[0], baudrate=115200, timeout=5)

    ts = time.time()

    readings = {
        "led1": [],
        "led2": [],
        "led3": [],
        "led4": [],
        "led5": [],
        "led6": [],
        "led7": [],
        "led8": [],
    }

    value = write_read(arduino, "ping")
    print(f"Ping:\n{value}")

    for i in range(NUM_READINGS):

        try:
            data = write_read(arduino, "scan")
            print(f"Scan:\n{data}")   

            if sys.getsizeof(data) > 33: # check against empty

                # Decode byte object
                data_decoded = data.decode('utf-8')

                # convert to dict
                data_dict = json.loads(data_decoded)

                # store readings
                readings['led1'].append(data_dict['led1'])
                readings['led2'].append(data_dict['led2'])
                readings['led3'].append(data_dict['led3'])
                readings['led4'].append(data_dict['led4'])
                readings['led5'].append(data_dict['led5'])
                readings['led6'].append(data_dict['led6'])
                readings['led7'].append(data_dict['led7'])
                readings['led8'].append(data_dict['led8'])
        except:
            print("ERROR: read")
            continue

    if not os.path.exists('data'):
        os.makedirs('data')

    df = pd.DataFrame.from_dict(readings)
    df.to_csv(f"./data/{FILENAME}")
    
    te = time.time()
    print(f"\nElapsed time: {te - ts}s")

if __name__ == "__main__":
    main()

