# 3dentification

A 3D printing plastics specific identification system combining Near-Infrared (NIR) Spectroscopy and computer vision. 

## Application

Ensure docker daemon is running: 
```
sudo systemctl start docker
```

Run the entire application (add `--build` to rebuild):
```
docker compose up -d
```

Take down:
```
docker compose down
```

### Run Kafka Instance 

Start:
```
docker compose -f ./utils/kafka.yaml up -d
```

Take down containers:
```
docker compose -f ./utils/kafka.yaml down
```
