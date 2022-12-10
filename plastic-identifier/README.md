# plastic-identification

Using NIR Spectroscopy to identify plastics. 

# Protobuf 

### Install on Linux

```
sudo apt install -y protobuf-compiler
```

### Compile

```
protoc --python_out=. ./proto/firmware.proto
```