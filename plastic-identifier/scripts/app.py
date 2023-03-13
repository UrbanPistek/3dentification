import tkinter
import random
import tkinter.messagebox
import customtkinter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# custom modules
from data_collection import get_scan
from lib.postprocess import SpectraGen

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

calibration_df = None
scan_df = None

fig = plt.Figure(figsize=(12, 7), dpi=100)
ax = fig.add_subplot(111)

# Leds used
LEDS = [850, 940, 1050, 890, 1300, 880, 1550, 1650]
Spectra = SpectraGen(led_wavelengths=LEDS)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("DNIR Scanner")
        self.geometry(f"{1200}x{900}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Controls", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # calibrate
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Calibrate", command=self.calibrate)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        # scan
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Scan", command=self.scan)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        # clear
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Clear", command=self.clear_plot)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        # Utility features
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # Create plot frame
        self.plot_frame = customtkinter.CTkFrame(self, width=600, height=700, corner_radius=2)
        self.plot_frame.grid(row=0, column=1, padx=(20, 20), pady=(10, 10), sticky="nsew")

        # add canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Text frame for data dumping 
        self.textbox = customtkinter.CTkTextbox(self, width=600, height=200)
        self.textbox.grid(row=1, column=1, padx=(20, 20), pady=(10, 10), sticky="nsew")
        
        # create scan info frame
        self.info_frame = customtkinter.CTkFrame(self)
        self.info_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.info_frame_group = customtkinter.CTkLabel(master=self.info_frame, text="Info:")
        self.info_frame_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")

        # create calibration info frame
        self.calibration_info = customtkinter.CTkFrame(self)
        self.calibration_info.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.calibration_info_group = customtkinter.CTkLabel(master=self.calibration_info, text="Calibration Info:")
        self.calibration_info_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        # Data processing configurations
        self.is_calibrated = False

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def scan(self):

        if self.is_calibrated:

            # Generate some random data to plot
            scan_df = get_scan()
            Spectra.add_measurements(scan_df)
            ys = Spectra.filtered_spectra()

            # zip & sort for nice plot
            zipped = list(zip(LEDS, ys))
            sorted_zipped = sorted(zipped, key=lambda x: x[0])
            xs, ys = zip(*sorted_zipped)

            # Clear the plot and plot the data
            ax.clear()
            ax.plot(xs, ys, label="Spectra", ls='--', marker='x', ms=7)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.set_title('Scan')
            ax.legend()
            self.canvas.draw()

        else:
            self.textbox.delete("0.0", "end")  # delete all text
            self.textbox.insert("0.0", "Please calibrate...")

    # Define the clear function
    def clear_plot(self):
        # Clear the plot
        ax.clear()
        self.canvas.draw()

    # Define the get function
    def calibrate(self):

        calibration_df = get_scan()
        
        # Remove noise and add calibration
        cali = Spectra.subtract_noise(df=calibration_df)
        Spectra.add_calibration_values(cali)

        self.textbox.delete("0.0", "end")  # delete all text
        self.textbox.insert("0.0", str(calibration_df))

        # update
        self.is_calibrated = True

if __name__ == "__main__":
    app = App()
    app.mainloop()