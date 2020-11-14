import cv2
import mss
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
from six import BytesIO
import uuid
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import datetime
import pyautogui 
import keyboard
import mouse
import scipy.cluster
import time

monitor = {"top": 562, "left": 947, "width": 25, "height": 25}
monitor["top"]=int((1080/2)-(monitor['width']/2))
monitor["left"]=int((1920/2)-(monitor['height']/2))
sct = mss.mss()
a = datetime.datetime.now()
percentage = 0.2

def load_image_into_numpy_array(image, is_file=False):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    if is_file:
        img_data = tf.io.gfile.GFile(image, 'rb').read()
        image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


while (True):
    img = np.array(sct.grab(monitor))
    # # to ger real color we do this:

    # frame = cv2.resize(frame, (640, 360))
    kernel = np.ones((30, 30), np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 150, 90])
    upper_red = np.array([180, 255, 180])

    mask = cv2.inRange(img, lower_red, upper_red)
    opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

   # median = np.ndarray.tolist(cv2.medianBlur(opening, 5))
    median = cv2.medianBlur(opening, 5)

    
    non_zero = np.array(np.nonzero(median)).flatten()
    #coords = list(map(list, zip(non_zero[0], non_zero[1])))
    b = datetime.datetime.now()
    c = b - a
    #print("got coords", c)
    a = b
    mid_x = int(monitor['width']/2)
    mid_y = int(monitor['height']/2)
    
    print("loop")
    if len(non_zero)>(monitor['width'] * monitor['height'] * percentage) and keyboard.is_pressed('alt'):
      #print(median)
      keyboard.press_and_release('u')
      time.sleep(0.125)
      print("click")

      b = datetime.datetime.now()
      c = b - a
      #print(c)
      a = b
      # cv2.imshow("title", opening)
    cv2.imshow("title", median)
    # cv2.imshow("title", blank_array)
    cv2.imshow("title2", img)
    # try:  # used try so that if user pressed other than the given key error will not be shown
    #     if keyboard.is_pressed('i') and abs(x_move) < 30 and abs(y_move) < 30:  # if key 'q' is pressed 
    #         print('You Pressed A Key!')
    #         keyboard.press_and_release('u')
    #         #break  # finishing the loop
    # except:
    #     break  
    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(25) & 0xFF == ord("s"):
        #119 = w
        #115 = s
        detection_id = str(uuid.uuid1())
        #pyautogui.moveRel(x_move, y_move, duration = 1) 
        #cv2.imwrite("output/manual_saves/" + detection_id + ".png", save_image)
