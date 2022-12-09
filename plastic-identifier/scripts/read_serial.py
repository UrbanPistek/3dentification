# Importing Libraries
import serial
import time

from lib.utils import get_serial_ports

def write_read(dev: serial.Serial, msg):
    dev.write(bytes(msg, 'utf-8'))
    
    time.sleep(0.5)
    data = dev.readline()
    
    return data

def main():

    # Print out all availible ports
    ports = get_serial_ports()
    print(f"Availible Ports: \n{ports}\n")

    arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)

    messages = ["ping", "adc", "scan"]
    for msg in messages:
        print(f"writing: {msg}")
        value = write_read(arduino, msg)
        print(f"recieved: {value}")

if __name__ == "__main__":
    main()