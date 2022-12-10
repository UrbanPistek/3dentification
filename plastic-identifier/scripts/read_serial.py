# Importing Libraries
import sys
import serial
import time
import json
import argparse

from lib.utils import get_serial_ports

def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser.parse_args:
    parser.add_argument(
        "-p",
        "--ping", 
        action='store_true',
        help='Ping Arduino')
    parser.add_argument(
        "-a",
        "--adc", 
        action='store_true',
        help='Read ADC value from Arduino')
    parser.add_argument(
        "-s",
        "--scan", 
        action='store_true',
        help='Read scan from Arduino')
    parser.add_argument(
        "-f",
        "--full", 
        action='store_true',
        help='Send all commands to the Arduino')

    return parser.parse_args()

def write_read(dev: serial.Serial, msg: str):
    dev.write(bytes(msg, 'utf-8'))
    
    time.sleep(0.5)
    data = dev.readline()
    
    return data

def write_read_blocking(dev: serial.Serial, msg: str, num_bytes=128):
    dev.write(bytes(msg, 'utf-8'))
    
    # Read until a specified number of bytes is recevied
    data = dev.read(size=num_bytes)

    return data

def main():

    parser = argparse.ArgumentParser()
    args = get_args(parser)

    # Print out all availible ports
    ports = get_serial_ports()
    print(f"Availible Ports: \n{ports}\n")
    print(f"Reading from port: \n{ports[0]}\n")

    # Using long timeout to wait for scan data
    arduino = serial.Serial(port=ports[0], baudrate=115200, timeout=10)

    # wakeup signal 
    value = write_read(arduino, "ping")
    print(f"Ping:\n{value}")

    if args.ping:
        value = write_read(arduino, "ping")
        print(f"Ping:\n{value}")

    elif args.adc:
        value = write_read(arduino, "adc")
        print(f"Adc:\n{value}")

    elif args.scan:
        ts = time.time()

        data = write_read_blocking(arduino, "scan", num_bytes=116)
        print(f"Scan:\n{data}")   
        print(f"size: {sys.getsizeof(data)}")     
        
        if sys.getsizeof(data) > 33: # check against empty
            te = time.time()
            print(f"\nElapsed time: {te - ts}s")

            # Decode byte object
            data_decoded = data.decode('utf-8')
            print(f"Decoded:\n{data_decoded}")
            print(f"type:\n{type(data_decoded)}")

            # convert to dict
            data_dict = json.loads(data_decoded)
            print(f"Dict:\n{data_dict}")
            print(f"Dict['led1']: {data_dict['led1']}")
            print(f"type:\n{type(data_dict)}")

    elif args.full:
        messages = ["ping", "adc", "scan"]
        for msg in messages:
            print(f"writing: {msg}")
            value = write_read(arduino, msg)
            print(f"recieved: {value}")

if __name__ == "__main__":
    main()