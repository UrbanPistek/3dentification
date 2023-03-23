import os
import tkinter
import random
import pickle
import tkinter.messagebox
import cv2
import copy
import webcolors
import customtkinter
import pandas as pd
import numpy as np
import PIL.Image, PIL.ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# custom modules
from data_collection import get_scan
from lib.postprocess import SpectraGen
from lib.colour import classify_color, get_colour_name

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

calibration_df = None
scan_df = None

fig = plt.Figure(figsize=(12, 7), dpi=100)
ax = fig.add_subplot(111)

cfig = plt.Figure(figsize=(1, 1), dpi=100)
cax = cfig.add_subplot(111)

# Leds used
LEDS = [850, 940, 1050, 890, 1300, 880, 1550, 1650]
Spectra = SpectraGen(led_wavelengths=LEDS)

labels = {
    "0": "abs",
    "1": "pla",
    "2": "empty",
    "3": "non_plastic",
    "4": "petg",
    "5": "plastic",
}

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("DNIR Scanner")
        self.geometry(f"{1600}x{900}")

        # configure grid layout (4x4)
        self.grid_columnconfigure((1, 2), weight=1)
        self.grid_rowconfigure((0, 2), weight=1)
        self.grid_rowconfigure((1), weight=0)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

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

        # inference
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Inference", command=self.inference)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        
        # get colour
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Get Colour", command=self.snapshot)
        self.sidebar_button_5.grid(row=5, column=0, padx=20, pady=10)

        # clear text box
        self.sidebar_button_6 = customtkinter.CTkButton(self.sidebar_frame, text="Clear Text", command=self.clear_text)
        self.sidebar_button_6.grid(row=6, column=0, padx=20, pady=10)

        # Utility features
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["50%", "75%", "80%", "90%", "100%", "110%", "120%", "150%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))

        # Create plot frame
        self.plot_frame = customtkinter.CTkFrame(self, width=700, height=400, corner_radius=2)
        self.plot_frame.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")

        # add canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Text frame for data dumping 
        self.textbox = customtkinter.CTkTextbox(self, width=300, height=300)
        self.textbox.grid(row=2, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")

        # Video capture
        self.vid = cv2.VideoCapture("/dev/video2")

        # create calibration info frame
        self.camera_feed = customtkinter.CTkFrame(self, width=400, height=300)
        self.camera_feed.grid(row=0, column=2, padx=(10, 20), pady=(10, 10), sticky="nsew")

        _, temp_frame = self.vid.read()
        temp_img = PIL.Image.fromarray(temp_frame)
        self.image = customtkinter.CTkImage(temp_img, size=(400, 300))
        self.camera_label_info = customtkinter.CTkLabel(self.camera_feed, text="Camera Feed:", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.camera_label_info.grid(row=0, column=0, padx=20, pady=10)
        self.camera_label = customtkinter.CTkLabel(self.camera_feed, text="", image=self.image)
        self.camera_label.grid(row=1, column=0, padx=20, pady=10)
        self.delay = 5
        self.update()

        # create calibration info frame
        self.calibration_info = customtkinter.CTkFrame(self, width=400, height=300)
        self.calibration_info.grid(row=2, column=2, padx=(10, 20), pady=(10, 10), sticky="nsew")
        self.colour_canvas = FigureCanvasTkAgg(cfig, master=self.calibration_info)
        self.colour_canvas.draw()
        self.colour_canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        # Data processing configurations
        self.is_calibrated = False
        self.is_scanned = False
        self.set_colour = False

        self.models = [
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model2_GradientBoostingClassifier_98_abs_pla_empty_2023_03_23.pickle",
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model2_GradientBoostingClassifier_99_abs_pla_empty_non_plastics_petg_plastics_2023_03_23.pickle",
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model2_QDA_0_abs_pla_empty_other_2023_03_23.pickle",
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model2_RandomForest_0_abs_pla_empty_2023_03_23.pickle",
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model2_RBFSVM_0_abs_pla_empty_non_plastics_petg_plastics_2023_03_23.pickle",
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model3_KNearestNeighbors_0_abs_pla_empty_other_2023_03_23.pickle",
            "/home/urban/urban/uw/fydp/3dentification/plastic-identifier/scripts/models/model3_HistogramGradientBoostingClassifier_0_abs_pla_empty_non_plastics_petg_plastics_2023_03_23.pickle",
        ]

        self.model_names_label = customtkinter.CTkLabel(self, text="Select Model:", anchor="w")
        self.model_names_label.grid(row=1, column=1, padx=20, pady=(1, 1, ), sticky="w")
        self.model_names = [str(os.path.basename(path)) for path in self.models if os.path.splitext(path)[1] == '.pickle']
        self.models_optionemenu = customtkinter.CTkOptionMenu(self, values=self.model_names, command=self.change_model)
        self.models_optionemenu.grid(row=1, column=1, padx=100, pady=(1, 1), sticky="w")

        self.model = self.models[0]
        self.clf = pickle.load(open(self.model, "rb"))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def change_model(self, new_model: str):
        for model_path in self.models:
            if new_model in model_path:
                self.model = model_path
                self.clf = pickle.load(open(self.model, "rb"))
                self.textbox.insert("end", f"\nLoaded Model: {new_model}")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get the height and width of the frame
            height, width = frame.shape[:2]

            # Calculate the coordinates of the center of the frame
            center_x = int(width / 2)
            center_y = int(height / 2)

            # Calculate the coordinates of the top-left and bottom-right corners of the square
            square_size = 75
            square_tl = (center_x - int(square_size / 2), center_y - int(square_size / 2))
            square_br = (center_x + int(square_size / 2), center_y + int(square_size / 2))

            # Draw the square overlay on the frame
            cv2.rectangle(frame, square_tl, square_br, (0, 0, 0), 2)

            img = PIL.Image.fromarray(frame)
            self.camera_label.imgtk = customtkinter.CTkImage(img, size=(400, 300))
            self.camera_label.configure(image=customtkinter.CTkImage(img, size=(400, 300)))

        self.after(self.delay, self.update)

    def snapshot(self):

        frames = []
        for i in range(5):

            _, frame = self.vid.read()
            n = frame.shape[0]
            m = frame.shape[1]
            dim = 75
            start_row = int((n-dim)/2)
            end_row = start_row + dim
            start_col = int((m-dim)/2)
            end_col = start_col + dim

            center_sample = copy.deepcopy(frame[start_row:end_row, start_col:end_col])
            frames.append(center_sample)

        frames = np.asarray(frames, dtype=float)
        mean = classify_color(frames)

        input_vals = (mean[2], mean[1], mean[0])
        _, closest_name = get_colour_name(input_vals)
        mean_colour = np.asarray(input_vals, dtype=int)
        self.textbox.insert("end", f"\nColour [rgb]: {closest_name} " + str(mean_colour))

        cax.clear()
        cax.set_title(f"Detected Colour: {closest_name}")
        cax.imshow([[mean_colour]])
        self.colour_canvas.draw()

        self.colour_values = mean_colour
        self.set_colour = True

    def scan(self):

        if self.is_calibrated:

            # Generate some random data to plot
            scan_df = get_scan()
            Spectra.add_measurements(scan_df)
            ys = Spectra.filtered_spectra()

            # store
            self.spectra_vals = ys
            self.is_scanned = True

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

            self.textbox.insert("end", "\n" + "=> Scan Complete!")

        else:
            self.textbox.insert("end", "\nPlease calibrate...")

    def clear_text(self):
        self.textbox.delete("0.0", "end")  # delete all text

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

        self.textbox.insert("end", "\n" + str(calibration_df))

        # update
        self.is_calibrated = True

    def inference(self):

        if self.is_scanned and self.set_colour:
            
            # fix dimensions for the model input
            input = self.spectra_vals
            ratios_vec = Spectra.create_ratios_vector()

            # normalize ratios
            max_ratio = max(ratios_vec)
            min_ratio = min(ratios_vec)
            for i in range(len(ratios_vec)):
                if (max_ratio - min_ratio) == 0:
                    ratios_vec[i] = 0
                elif ratios_vec[i] == 0:
                    ratios_vec[i] = 0
                else:
                    ratios_vec[i] = (ratios_vec[i] - min_ratio) / (max_ratio - min_ratio) # MinMax scaling
            input = np.concatenate((input, ratios_vec), axis=0)

            # normalize colour
            rgb = self.colour_values
            rgb = (rgb) / (255) # MinMax scaling
            input = np.concatenate((input, rgb), axis=0)
            input = input.reshape(1, -1)

            # make prediction
            res = self.clf.predict(input)
            
            if ("other" in self.model) and (res[0] == 3):
                self.textbox.insert("end", "\n" + "other")
            else: 
                self.textbox.insert("end", "\n" + labels[str(res[0])])

        else:
            self.textbox.insert("end", "\n" + "Please scan and/or get colour...")

if __name__ == "__main__":
    app = App()
    app.mainloop()