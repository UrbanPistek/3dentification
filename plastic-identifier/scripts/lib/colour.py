# OpenCV color detection
import cv2
import copy
import webcolors
import numpy as np

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def mean_colour(frame):
    average = frame.mean(axis=0).mean(axis=0)
    colors = (average[2], average[1], average[0])

    return average, colors

def dominant_color(frame):

    pixels = np.float32(frame.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    actual_name, closest_name = get_colour_name(dominant)

    return actual_name, closest_name

def classify_color(frames: np.ndarray):

    # get mean
    frame_avgs = []
    for frame in frames:
        stm = frame.mean(axis=0)
        ftm = stm.mean(axis=0)
        frame_avgs.append(ftm)

    frame_avgs = np.asarray(frame_avgs, dtype=float)
    # get mean
    mean = frame_avgs.mean(axis=0)

    return mean