import time
import cv2
import mss
import numpy as np
from PIL import Image
import object_tracking

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 240, "left": 2580, "width": 600, "height": 600}
# Set monitor size to capture
mon = (0, 40, 800, 640)


def screen_recordMSS():
    global fps, start_time, monitor
    while(True):
        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        # # to ger real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        frame_detection = object_tracking.detect(img, from_file=False)
        frame_detection.trim_by_score_threshold(0.8)
        labeled_output = object_tracking.pipeline(frame_detection.get_boxes(), frame_detection.get_image())

        cv2.imshow(title, labeled_output)
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


# def run_flow():
#     while(True):
#         ret, frame = cap.read()
#         frame_detection = detect(frame, from_file=False)
#         frame_detection.trim_by_score_threshold(0.8)
#         labeled_output = pipeline(frame_detection.get_boxes(), frame_detection.get_image())
#         cv2.imshow('frame', labeled_output)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
screen_recordMSS()

