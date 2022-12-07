# Importing Libraries
import serial
import time

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)

def write_read():
    arduino.write(bytes("scan", 'utf-8'))
    
    time.sleep(0.5)
    data = arduino.readline()
    
    return data

value = write_read()
print(f"recieved: {value}")