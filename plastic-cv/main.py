# OpenCV object and color detection
import cv2
import copy
import webcolors
import numpy as np

from colour import get_colour_name

# Store globally for now
vid = cv2.VideoCapture(0)

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
    smin = (mean[0:3] - 15) # Sub 5
    smax = (mean[0:3] + 15) # Add 5

    minRange = tuple(smin.tolist())
    maxRange = tuple(smax.tolist())

    mask = cv2.inRange(frame, minRange, maxRange)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # print(f"masked frame: {masked_frame.shape}")

    # remove zeros (black)
    filt_frame = masked_frame.mean(axis=2)

    # print(f"filtered masked frame: {filt_frame.shape}")

    # get non-zero values
    non_zero = np.nonzero(filt_frame)
    # print(f"non_zero: {len(non_zero[0])}, {len(non_zero[1])}")

    final = masked_frame[non_zero]
    # print(f"final: {final.shape}")
    # print(f"final mean: {final.mean(axis=0)}")
    # print(f"final min: {final.min(axis=0)}")    
    # print(f"final max: {final.max(axis=0)}")

    mean_flt = final.mean(axis=0)

    input_vals = (mean_flt[2], mean_flt[1], mean_flt[0])
    actual_name, closest_name = get_colour_name(input_vals)
    print ("mean actual colour name:", actual_name)
    print ("mean closest colour name:", closest_name)

    return masked_frame

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
            cv2.imshow('frame', frame) # display video feed

            # collect set of frames
            data = copy.deepcopy(frame)
            frames.append(data)
            if len(frames) > 25: 
                print("calculating...")
                frames = np.asarray(frames, dtype=float)
                mask = classify_color(frames)
                cv2.imshow('mask', mask)
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
