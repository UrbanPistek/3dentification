# OpenCV object and color detection
import cv2
import copy
import webcolors
import numpy as np
import threading

from util.colour import get_colour_name

# Store globally for now
# vid = cv2.VideoCapture(0) # webcam
vid = cv2.VideoCapture("/dev/video2") # usb cam

# Get baseline values
def setup():

    # average our a few frames
    frame_avgs = []
    for i in range(0, 25): # use 250
        _ , frame = vid.read()
        stm = frame.mean(axis=0)
        ftm = stm.mean(axis=0)
        frame_avgs.append(ftm)

    frame_avgs = np.asarray(frame_avgs, dtype=float)

    # get mean
    mean = frame_avgs.mean(axis=0)

    # get standard deviation for all channels
    stddevs = np.std(frame_avgs, axis = 0)

    return mean, stddevs

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

    # Drop alpha channel and add tolerance
    smin = (mean[0:3] - 5) # Sub 5
    smax = (mean[0:3] + 5) # Add 5

    minRange = tuple(smin.tolist())
    maxRange = tuple(smax.tolist())

    # mask = cv2.inRange(frame, minRange, maxRange)
    # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    input_vals = (mean[2], mean[1], mean[0])
    actual_name, closest_name = get_colour_name(input_vals)
    print(f"mean: {mean}")
    print ("mean actual colour name:", actual_name)
    print ("mean closest colour name:", closest_name)

    return mean

def read_single_frame():

    _ , frame = vid.read()
    stm = frame.mean(axis=0)
    ftm = stm.mean(axis=0)

    print(f"frame: {frame.shape}")
    print(f"stm: {stm.shape}")
    print(f"ftm: {ftm.shape}")

    print(f"single: {ftm}")

def main():
    print("plastic-cv...")

    read_single_frame()

    # base_mean, base_std = setup()
    # print(f"setup() mean:{base_mean}, std:{base_std}")

    frames = []
    try:
        while True:

            # capturing the current frame
            _ , frame = vid.read()
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

            # Display the resulting frame
            cv2.imshow('Camera Feed', frame)

            # collect set of frames
            # Get a 10x10 sample of the center of the matrix
            n = frame.shape[0]
            m = frame.shape[1]
            dim = square_size
            start_row = int((n-dim)/2)
            end_row = start_row + dim
            start_col = int((m-dim)/2)
            end_col = start_col + dim

            center_sample = copy.deepcopy(frame[start_row:end_row, start_col:end_col])
            # data = copy.deepcopy(frame)

            frames.append(center_sample)
            if len(frames) > 25: 
                print("calculating...")
                frames = np.asarray(frames, dtype=float)
                
                mean = classify_color(frames)
                # mask = cv2.inRange(frame, lower, upper)
                # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                # cv2.imshow('colour detected', masked_frame)

                color = np.array([[[mean[2], mean[1], mean[1]]]], dtype=np.uint8)

                # Convert the color array to BGR format (required by OpenCV)
                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

                # Create a 25x25 black image
                image = np.zeros((25, 25, 3), dtype=np.uint8)

                # Fill the image with the desired color
                image[:] = color_bgr

                # Display the resulting image
                cv2.imshow('Color Plot', image)

                frames = []

            print(f"frames[{len(frames)}]")

            # press Q to stop
            if cv2.waitKey(1) & 0xFF == ord('Q'):
                break

        # release video capture
        vid.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\nexit...")

if __name__ == "__main__":
    main()
