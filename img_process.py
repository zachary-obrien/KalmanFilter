import cv2
import mss
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
from six import BytesIO

monitor = {"top": 240, "left": 2580, "width": 512, "height": 412}

sct = mss.mss()


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
    kernel = np.ones((15, 15), np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 150, 100])
    upper_red = np.array([180, 255, 180])

    mask = cv2.inRange(img, lower_red, upper_red)
    opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    median = cv2.medianBlur(opening, 15)

    cv2.imshow("title", median)

    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
