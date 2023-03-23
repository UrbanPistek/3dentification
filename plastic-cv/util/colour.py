import cv2
import argparse
import webcolors
import numpy as np

def get_args(parser):
    parser.add_argument(
        "-c",
        "--colours", 
        action='store_true',
        help='Run color filtering')

    return parser.parse_args()

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

    print(f"mean color: {average}")
    colors = (average[2], average[1], average[0])

    actual_name, closest_name = get_colour_name(colors)
    print ("mean actual colour name:", actual_name)
    print ("mean closest colour name:", closest_name)

    return average, colors

def dominant_color(frame):

    pixels = np.float32(frame.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    print(f"dominant color: {dominant}")
    actual_name, closest_name = get_colour_name(dominant)
    print ("dominant closest colour name:", closest_name)
    print ("dominant actual colour name:", actual_name)
    print ("dominant closest colour name:", closest_name)

def main():
    print("plastic-cv:colours...")

    parser = argparse.ArgumentParser()
    args = get_args(parser)

    if args.colours:
        vid = cv2.VideoCapture(0)

        try:
            while True:

                # capturing the current frame
                _ , frame = vid.read()

                # display video feed
                cv2.imshow('frame', frame)
                
                m, c = mean_colour(frame)

                # Generate mask based on mean color
                stm = frame.mean(axis=0)
                ftm = stm.mean(axis=0)
                
                # Drop alpha channel and add tolerance
                smin = (ftm[0:3] - 15) # Sub 5
                smax = (ftm[0:3] + 15) # Add 5

                minRange = tuple(smin.tolist())
                maxRange = tuple(smax.tolist())

                print(f"min: {minRange}, max: {maxRange}")

                mask = cv2.inRange(frame, minRange, maxRange)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                cv2.imshow('masked_frame', masked_frame)

                # Perform estimates on filterd frame
                print(f"\nFiltered Frame:")
                m, c = mean_colour(masked_frame)               
                # dominant_color(frame)

                # press Q to stop
                if cv2.waitKey(1) & 0xFF == ord('Q'):
                    break

            # release video capture
            vid.release()

            # Closes all the frames
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\nexit...")

    else:
        print("Please specific mode of execution:\n")
        parser.print_usage()
        parser.print_help()

if __name__ == "__main__":
    main()