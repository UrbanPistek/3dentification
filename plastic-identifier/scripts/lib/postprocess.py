import copy
import pandas as pd
import numpy as np

def subtract_noise(df: pd.DataFrame) -> np.ndarray: 

    new_df = df.drop(['units'], axis=1)
    new_df = new_df.to_numpy()

    # Subtract out noise
    signal = new_df[0] - new_df[2]
    signal[signal < 0] = 0 # Any negative values from the subtraction get mapped to zero

    return signal

class SpectraGen: 

    def __init__(self, measurements: pd.DataFrame, led_wavelengths: np.ndarray, ref: np.ndarray = np.zeros(8), cal: np.ndarray = np.zeros(8)) -> None:
        
        # Store which LED wavelengths are being used
        self.led_wavelengths = led_wavelengths

        # Reference values to normalize data
        self.ref_values = ref

        # Calibration value - tracks LED scatter
        self.cal_values = cal

        # Measured Values
        self.measurements = measurements
    
    def add_reference_values(self, ref: np.ndarray):
        
        # Reference values to normalize data
        self.ref_values = ref

    def add_calibration_values(self, cal: np.ndarray):
        
        # Calibration value - tracks LED scatter
        self.cal_values = cal

    def get_variances(self) -> np.ndarray:
        pass

    def get_ambient_noise(self) -> np.ndarray:
        pass

    def filtered_spectra(self) -> np.ndarray:
        
        # Store processed values - copy initial values
        df = copy.deepcopy(self.measurements)

    def normalize(self):
        pass

