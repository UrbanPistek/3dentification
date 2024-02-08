# 3dentification

An identification system specific to 3D-printing plastics which combines Near-Infrared (NIR) Spectroscopy and computer vision (CV). See POSTER.png for a detailed visual representation of the project!

Goal: To reliably classify PLA, ABS, PETG, and other common materials used in 3D-printing so they can be sorted by material and colour.

## Motivation 

3D-printing is a growing industry, and with it comes a growing waste problem. 7.1 MILLION kg of 3D-printed thermoplastics are deposited in landfills every year in the USA alone. 

While 3D-printed plastic parts are very easy to recycle when they are reliably sorted, even a few specks of contaminations from other plastics can ruin the material properties of the recycled filament and make it impossible to reuse in a 3D-printer.

While solutions exist for industrial-scale 3D-printed waste, hobbyists, libraries, makerspaces, and other smaller users have no choice but to dispose of their test prints, failed prints, and support material in the garbage. 

There is one gap creating this persisting problem. While the technology to recycle 3D-printed thermoplastic waste exists, no technology exists to sort it reliably. 3D-printed waste can be any shape, colour, and density, and aren't classified with numeric triangle recycling symbols ("resin identification codes"). As a result, municipal recyclers can't sort out this waste stream, and it end up in the landfill.

Our interdisciplinary 4th year engineering "capstone design" team (2 mechatronics eng, 1 chemical eng, 1 environmental eng) set out to tackle this problem. We partnered with [3cycle](https://3cycle.ca), a local startup and social entreprise working on establishing waste collection and recycling for small-scale 3D-printer users. 3cycle graciously helped fund this project -- many thanks!

The code in this repository runs the NIR spectroscopy plastic material identification sensor we built based on the open-source project [Plastic Scanner](https://github.com/Plastic-Scanner), the CV sensor for colour identification, and the user interface which controls it all.

## Technologies Used

### Near-Infrared (NIR) Spectroscopy

This is the science at the core of the plastic scanner. Light in the near end of the infrared spectrum (from .7 to 1.4 micrometers in wavelength) is emitted by LEDs, reflected off the material sample, and measured by a photodiode. Different materials absorb different amounts of each wavelength of light. By measuring 8 different wavelengths in sequence at points of the spectrum where our materials of interest (PLA, ABS, etc) differ significantly, and subtracting signal noise identified with a calibration scan, we can determine with confidence which material is being sensed. 

### Computer Vision

Using a small USB camera, a live video feed captures the colour of the sample by taking the average RGB colour values of the pixels onscreen, minus a calibration image when no sample was present. 

## Machine Learning

By collecting samples of known material, scanning them, and labelling that data with the material of the sample, we created a dataset specific to our sensor which we used to train a software model how to classify new, unlabelled samples. By the time the poster was made, the software model could accurately distinguish between PLA and ABS 97% of the time, but by collecting more samples and making our dataset larger, that result reached over 99%. 

## Languages and Technologies
Python, C++, Rust, Arduino, Docker, PlatformIO

Credits:
Urban Pistek
Julia Baribeau
Mohammed Abayazeed
Maria Fraser-Semenoff 

Special thanks to 3cycle founder Jason Amri, the Plastic Scanner team, Velocity Science, advisors Tizazu Mekonnen, Simar Saini, Oscar Nespoli, and Binh Minh Trinh from the Polymers Lab, and Jeff at Accelerated Systems Inc. for making this project possible.

## Notes on Running the Application

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
