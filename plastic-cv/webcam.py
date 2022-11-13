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
    parser.add_argument(
        "-o",
        "--objects", 
        action='store_true',
        help='Run object detection')

    return parser.parse_args()

def colour_detection(frame):

    # setting values for base colors
    b = frame[:, :, :1]
    g = frame[:, :, 1:2]
    r = frame[:, :, 2:]

    # computing the mean
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    # displaying the most prominent color
    if (b_mean > g_mean and b_mean > r_mean):
        print("Blue")
    if (g_mean > r_mean and g_mean > b_mean):
        print("Green")
    else:
        print("Red")

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define blue color range
    light_blue = np.array([110,50,50])
    dark_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, light_blue, dark_blue)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(frame,frame, mask= mask)

    # Display the frame, saved in the file   
    cv2.imshow('output',output)

def face_detect(vid, faceCascade):
    # capturing the current frame
    _, frame = vid.read()

    # converting image from color to grayscale 
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    # 1.3 = scale factor, 5 = minimum neighbor can be detected
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    print(f"face detected: {faces}") 

    # drawing bounding box around face
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,   0), 3)

    # displaying image with bounding box
    cv2.imshow('face_detect', frame)

def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = None
    return closest_name

def mean_colour(frame):
    average = frame.mean(axis=0).mean(axis=0)

    print(f"mean color: {average}")
    closest_name = get_colour_name(average)
    print ("mean closest colour name:", closest_name)

def dominant_color(frame):

    pixels = np.float32(frame.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    print(f"dominant color: {dominant}")
    closest_name = get_colour_name(dominant)
    print ("dominant closest colour name:", closest_name)


def main():
    print("plastic-cv:webcam...")

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
                
                # colour_detection(vid)
                mean_colour(frame)
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

    elif args.objects:

        faceCascade = cv2.CascadeClassifier()
        if not faceCascade.load(cv2.samples.findFile("./data/haarcascade_frontalface_default.xml")):
            print('--(!)Error loading face cascade')
            exit(0)
            
        vid = cv2.VideoCapture(0)

        try:
            while True:
                
                face_detect(vid, faceCascade)
                
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