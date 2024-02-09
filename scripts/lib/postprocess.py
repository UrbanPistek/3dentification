import copy
import pandas as pd
import numpy as np

class SpectraGen: 

    def __init__(self, measurements: pd.DataFrame = pd.DataFrame(), led_wavelengths: np.ndarray = np.zeros(8), ref: np.ndarray = np.zeros(8), cal: np.ndarray = np.zeros(8)) -> None:
        
        # Store which LED wavelengths are being used
        self.led_wavelengths = led_wavelengths

        # Reference values to normalize data
        self.ref_values = ref

        # Calibration value - tracks LED scatter
        self.cal_values = cal

        # Measured Values
        self.measurements = measurements
    
    def add_reference_values(self, ref: np.ndarray) -> None:
        
        # Reference values to normalize data
        self.ref_values = ref

    def add_calibration_values(self, cal: np.ndarray) -> None:
        
        # Calibration value - tracks LED scatter
        self.cal_values = cal

    def add_measurements(self, mes: pd.DataFrame) -> None:

        # Add measurements after the fact
        self.measurements = mes

    def add_led_wavelengths(self, leds: np.ndarray) -> None:
        
        # Add wavelengths of each LED used
        self.led_wavelengths = leds

    def get_values(self) -> np.ndarray:
        
        # Variances are located in the second row
        return self.measurements.to_numpy()[0][1:]

    def get_variances(self) -> np.ndarray:
        
        # Variances are located in the second row
        return self.measurements.to_numpy()[1][1:]

    def get_ambient_noise(self) -> np.ndarray:
        
        # Variances are located in the second row
        return self.measurements.to_numpy()[2][1:]

    def filtered_spectra(self) -> np.ndarray:
        
        # Subtract out noise 
        denoised = self.subtract_noise(self.measurements)

        # Remove calibration which is the LED scatter noise
        spectra = denoised - self.cal_values

        # Any negative values get mapped to zero
        spectra[spectra < 0] = 0

        self.raw_spectra = spectra
        return spectra
    
    def create_ratios_vector(self) -> np.ndarray:

        dim = len(self.raw_spectra)
        ratio_matrix = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if self.raw_spectra[j] == 0:
                    ratio_matrix[i,j] = 0
                else:
                    ratio_matrix[i,j] = self.raw_spectra[i] / self.raw_spectra[j]

        # Flatten the matrix into a 1D array using the ravel() method
        ratio_vector = ratio_matrix.ravel()
        self.ratio_vector = ratio_vector
        return ratio_vector

    def normalize(self):
       
        # apply min/max normalization to spectra
        min_val = 0 # min(self.ref_values)
        max_val = max(self.ref_values)

        normalized_spectra = (self.raw_spectra - min_val) / (max_val - min_val)

        self.normalized_spectra = normalized_spectra
        return normalized_spectra

    def subtract_noise(self, df: pd.DataFrame) -> np.ndarray: 

        new_df = df.drop(['units'], axis=1)
        new_df = new_df.to_numpy()

        # Subtract out noise
        signal = new_df[0] - new_df[2]
        signal[signal < 0] = 0 # Any negative values from the subtraction get mapped to zero

        return signal
    
    def display(self) -> None:
        """
        Displays the current object configuration
        """
        print(f"\nLeds:\n{self.led_wavelengths}")
        print(f"\nReference Values:\n{self.ref_values}")
        print(f"\nCalibration Values:\n{self.cal_values}")
        print(f"\nMeasurements:\n{self.measurements}")
