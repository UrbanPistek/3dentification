"""DNIR Module Server"""
import time
import random
from flask import Flask
from matplotlib import pyplot as plt

import tkinter as tk
import sv_ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# custom modules
from data_collection import get_scan
from lib.postprocess import SpectraGen

calibration_df = None
scan_df = None

# Leds used
LEDS = [850, 940, 1050, 890, 1300, 880, 1550, 1650]
Spectra = SpectraGen(led_wavelengths=LEDS)

# Create the GUI window
root = tk.Tk()
root.title("Data Plotter")
root.geometry("1200x900")

# Create the plot and add it to the window
fig = plt.Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Define the plot function
def plot_data():
    # Generate some random data to plot
    x = [i for i in range(10)]
    y = [random.randint(1, 10) for i in range(10)]
    
    # Clear the plot and plot the data
    ax.clear()
    ax.plot(x, y)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Data Plot')
    canvas.draw()

# Define the clear function
def clear_data():
    # Clear the plot
    ax.clear()
    canvas.draw()

# Define the get function
def get_data():
    # Retrieve the data from the plot
    pass

# Create the button frame and buttons
button_frame = tk.Frame(root)
plot_button = tk.Button(button_frame, text="Plot", command=plot_data)
clear_button = tk.Button(button_frame, text="Clear", command=clear_data)
get_button = tk.Button(button_frame, text="Get", command=get_data)

# Add the buttons to the frame
plot_button.pack(side=tk.LEFT, padx=5)
clear_button.pack(side=tk.LEFT, padx=5)
get_button.pack(side=tk.LEFT, padx=5)

# Add the frame to the window
button_frame.pack(side=tk.BOTTOM, padx=5, pady=5)

sidebar = tk.Frame(root, bg="white", width=200, height=900)
sidebar.pack(side=tk.LEFT, fill=tk.BOTH)

# Add some data and text to the sidebar
sidebar_label = tk.Label(sidebar, text="Sidebar Data", font=("Arial", 12))
sidebar_label.pack(pady=20)

# set theme
sv_ttk.set_theme("light")

# Run the GUI
root.mainloop()
