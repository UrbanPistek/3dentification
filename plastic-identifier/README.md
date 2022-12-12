# plastic-identification

Using NIR Spectroscopy to identify plastics. 

## Arduino Communication
Communicate with arduino over serial: 
[PySerial](https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial.read)

# Protobuf 

### Install on Linux

```
sudo apt install -y protobuf-compiler
```

### Compile

```
protoc --python_out=. ./proto/firmware.proto
```