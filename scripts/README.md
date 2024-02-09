# Scripts

### Running the GUI Application

Firstly, you'll need to check that the right video device is selected, on linux you can see availible video with the following command:
```
ls /dev | grep video
```

Then updated the following line in `app.py`:
```
self.vid = cv2.VideoCapture("/dev/video0")
```

You can now run the application, make sure you are running it from the `scripts` directory.
```
python app.py
```

### Model Development

Collected DNIR Data from our Board is availible under the `./data` folder, **dataset4** is the most complete and comprehensive dataset.

You can use the script `scripts/generation.py` to generate synthetic data based on the ground truth data collected.

```
python -u generation.py 
```

The script `scripts/train.py` was used to train various models, validate models, generate confusion matrices, estimate Mean Absolute Error and even opitimize models using random halving grid search.

Run using `scripts/train.py -e` followed by the appropiate cli flags:

You should be able to run the script as is to train on some mock data as follows: `python -u train.py -e -m -t `

```
usage: train.py [-h] [-v] [-s] [-o] [-t] [-r] [-c] [-m] [-e]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Run training in verbose, showing confusion matrices and outputs
  -s, --save            Save model
  -o, --optimize        Run optimization on a set of models
  -t, --train           Train a set of models
  -r, --random-shuffle  Randomly shuffle training/testing data
  -c, --check-overfitting
                        Check if models are overfitting on the data
  -m, --mock-data       Use synthetic data for training
  -e, --execute         Run actions in the script

```

The following script was used to collect serial data from the arduino over a serial usb connection:

```
scripts/data_collection.py
```
