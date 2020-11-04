import cv2
import numpy as np
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
from pykalman import KalmanFilter
from matplotlib.patches import Ellipse
import PIL
from PIL import Image
import scipy
from scipy import signal
import matplotlib.patches as patches
import pandas as pd
import time
import sys
import os
from scipy.special import softmax

VIDEO_FILE_NAME = 'singleball.mov'
TEMPLATE_DERIVED_FRAME = 15
TEMPLATE_CROP_COORDS_FOR_FRAME_15 = (73, 251, 25, 22)
TEMPLATE_FILE_PNG = 'ball_template.png'
TEMPLATE_FILE_NPY = 'ball_template.npy'
TEMPLATE_MATCH_METHODS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                          'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
BEST_MATCH_METHOD = ['cv2.TM_CCORR_NORMED']
SAVE_DIRECTORY = 'C:/Users/Cooper/Documents/GitHub/TumorTracking_2nd_ext/Kalman-2/template_comparison_frames/'
PNG_DIRECTORY = 'C:/Users/Cooper/Documents/GitHub/TumorTracking_2nd_ext/Kalman-2/template_match_frames/'
OUTPUT_DIRECTORY = 'C:/Users/Cooper/Documents/GitHub/TumorTracking_2nd_ext/Kalman-2/template_match_frames/video'


def get_template_match_positions(target_frame, template_file_name, match_method):
    """Return a numpy array of where template matches the target frame"""

    # evals the string function for the template matching algorithm
    match_method = eval(match_method)

    template_frame = np.load(template_file_name)

    result = cv2.matchTemplate(
        target_frame.astype(np.float32),
        template_frame.astype(np.float32),
        match_method)

    # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    # _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    # matchLoc = maxLoc
    # cv2.rectangle(img, matchLoc, (matchLoc[0] + templ))

    _minVal, _maxVal, min_loc, max_loc = cv2.minMaxLoc(result, None)
    print(_minVal)
    print(_maxVal)
    print(min_loc)
    print(max_loc)

    return result


def get_max_and_loc(target_frame, template_file_name, match_method):
    """returns the max value and location of a template match"""
    # evals the string function for the template matching algorithm
    match_method = eval(match_method)

    # loads numpy array of template match
    template_frame = np.load(template_file_name)

    result = cv2.matchTemplate(
        target_frame.astype(np.float32),
        template_frame.astype(np.float32),
        match_method)

    _minVal, _maxVal, min_loc, max_loc = cv2.minMaxLoc(result, None)

    return _maxVal, max_loc


def get_template_match_png(target_frame, template_file_name, match_method):
    """Template matches a .png file of a template and a np array of a target frame. Converts arrays to Images first"""
    # template_frame = np.load(template_file_name)

    # evals the string function for the template matching algorithm
    match_method = eval(match_method)

    # reads the .png file into a cv2 image
    template_img = cv2.imread(template_file_name, cv2.IMREAD_COLOR)

    # saves the target_frame numpy array to a temp .png file
    target_img_temp = Image.fromarray(target_frame, 'RGB')
    target_img_temp.save('temp_target.png')

    # reads the temp .png target file into a cv2 image
    target_img = cv2.imread('temp_target.png', cv2.IMREAD_COLOR)

    # creates copy of target image to draw rectangle on
    img_display = target_img.copy()

    # templates matches the template image to the target image
    result = cv2.matchTemplate(target_img, template_img, match_method)

    # normalizes the match values
    # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    # finds the min and max of result
    _minVal, _maxVal, min_loc, max_loc = cv2.minMaxLoc(result, None)

    # sets match location to the maximum location value
    match_loc = max_loc

    # draws rectangles on the target frame
    cv2.rectangle(img_display, match_loc, (match_loc[0] + template_img.shape[0], match_loc[1] + template_img.shape[1]),
                  (0, 255, 0), 1, 8, 0)
    # cv2.rectangle(result, match_loc, (match_loc[0] + template_img.shape[0], match_loc[1] + template_img.shape[1]), (0,0,0), 2, 8, 0)

    # shows the image with a box around the matched template
    cv2.imshow('Matched Image', img_display)
    cv2.waitKey(0)


def get_frame_np_array(file_name, target_frame_number):
    """Return a numpy array of a specified frame from a video file"""
    capture_object = cv2.VideoCapture(file_name)
    capture_object.set(1, target_frame_number);
    ret, frame = capture_object.read()
    return frame


def show_frame_np_array(np_frame):
    """Show plot of numpy array frame"""
    plt.imshow(np_frame)
    plt.colorbar()
    plt.show()


def get_template_crop_from_frame(np_frame, x_origin, y_origin, x_width, y_height):
    """Return a cropped numpy array of a frame with origin coordinates and width/height"""
    # return np_frame[251:274, 73:99, :]
    return np_frame[y_origin:y_origin + y_height, x_origin:x_origin + x_width, :]


def save_np_array_as_image(np_array, image_name):
    """Save a png file of an numpy array"""
    im = Image.fromarray(np_array)
    im.save(image_name)


def save_np_array_as_npy(np_array, file_name):
    """Save a .npy file of an np array"""
    with open(file_name, 'wb') as f:
        np.save(f, np_array)


def template_match_use_all_algorithms(target_frame, template_file_name, process_flag, methods, frame_num):
    """Outputs the template match numpy array image, the png image with the bounding box
    and the name of the algorithm used to template match"""

    # reads the .png file into a cv2 image
    template_img = cv2.imread(template_file_name, cv2.IMREAD_COLOR)

    # saves the target_frame numpy array to a temp .png file
    target_img_temp = Image.fromarray(target_frame, 'RGB')
    target_img_temp.save('temp_target.png')

    # reads the temp .png target file into a cv2 image
    target_img = cv2.imread('temp_target.png', cv2.IMREAD_COLOR)

    # c = channels, w = width, h = height
    c, w, h = template_img.shape[::-1]

    for match_method in methods:
        # evaluates the string function for the template matching algorithm
        match_method = eval(match_method)

        # creates copy of target image to draw rectangle on
        img_display = target_img.copy()

        if process_flag == 'np_array':
            # reads template .npy file
            template_frame_np = np.load(TEMPLATE_FILE_NPY)

            # template matches off of np arrays
            result = cv2.matchTemplate(
                target_frame.astype(np.float32),
                template_frame_np.astype(np.float32),
                match_method)

            original_file_string = 'template match by: numpy\n'

        elif process_flag == 'png':
            # templates matches the template image to the target image
            result = cv2.matchTemplate(target_img, template_img, match_method)

            original_file_string = 'template match by: .png\n'

        # normalizes the match values to values between 0 and 1
        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

        # finds the min and max of result
        _minVal, _maxVal, min_loc, max_loc = cv2.minMaxLoc(result, None)
        print(_maxVal, max_loc)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # draw bounding box on image copy
        cv2.rectangle(img_display, top_left, bottom_right, (0, 255, 0), 2)

        # extract algorithm name
        method_string = TEMPLATE_MATCH_METHODS[match_method].split('.')[1]

        plt.subplot(121), plt.imshow(result)
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_display)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

        var_str = '\nmin value: {}\n max value: {}\n min loc: {}\n max loc: {}' \
            .format(_minVal, _maxVal, min_loc, max_loc)
        plt.suptitle('frame: ' + str(frame_num) + '\n' + original_file_string + method_string + var_str)
        # plt.savefig(PNG_DIRECTORY + str(frame_num) + '.png', format='png')
        plt.show()


def write_movie():
    """builds a mp4 video given a directory of .png files"""

    images = []
    for f in os.listdir(PNG_DIRECTORY):
        if f.endswith('.png'):
            images.append(f)

    num_images = []
    for img in images:
        new_image = img.split('.')[0]
        num_images.append(new_image)

    num_images.sort(key=int)

    sorted_images = []
    for img in num_images:
        sorted_images.append(img+'.png')

    # Determine the width and height from the first image
    image_path = os.path.join(PNG_DIRECTORY, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter('template_match.mp4', fourcc, 4, (width, height))

    for image in sorted_images:
        image_path = os.path.join(PNG_DIRECTORY, image)
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


def template_match_all_frames(file_name):
    """Template matches all frames in a video file"""
    capture_object = cv2.VideoCapture(file_name)

    if not capture_object.isOpened():
        print("Error opening video stream or file")
    frame_num = 0

    while capture_object.isOpened():
        ret, frame = capture_object.read()
        if frame is not None:
            template_match_use_all_algorithms(frame, TEMPLATE_FILE_PNG, 'np_array', BEST_MATCH_METHOD, frame_num)
            # max_and_loc = get_max_and_loc(frame, TEMPLATE_FILE_NPY, 'cv2.TM_CCORR_NORMED')
            # print(frame_num, max_and_loc)
            print(frame_num)
            frame_num += 1
            # time.sleep(.100)
        else:
            continue

    # When everything done, release the video capture object
    capture_object.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def get_max_and_max_loc_whole_file(file_name):
    """returns a dataframe containing frame #, max_value, max_loc"""
    capture_object = cv2.VideoCapture(file_name)

    if not capture_object.isOpened():
        print("Error opening video stream or file")

    frame_number = 0
    frame_vals = []
    max_vals = []
    max_x_locs = []
    max_y_locs = []

    while capture_object.isOpened():
        ret, frame = capture_object.read()
        if frame is not None:
            frame_vals.append(frame_number)

            max_and_loc = get_max_and_loc(frame, TEMPLATE_FILE_NPY, 'cv2.TM_CCORR_NORMED')

            max_vals.append(max_and_loc[0])
            max_x_locs.append(max_and_loc[1][0])
            max_y_locs.append(max_and_loc[1][1])

            frame_number += 1
        else:
            break

    df = pd.DataFrame(data={'frames': frame_vals, 'max_values': max_vals, 'x_coord': max_x_locs, 'y_coord': max_y_locs})

    # When everything done, release the video capture object
    capture_object.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return df


def plot_coords(data_frame):
    """plots the x,y coords of the tempalte matched dataframe"""
    # reads the .png file into a cv2 image
    template_img = cv2.imread(TEMPLATE_FILE_PNG, cv2.IMREAD_COLOR)
    # c = channels, w = width, h = height
    c, w, h = template_img.shape[::-1]

    # build graph of bounding boxes
    fig, ax = plt.subplots(360, 480)

def get_softmax_frame(target_frame):
    """Prints the softmax values of a numpy array from a video numpy frame"""
    softmax_frame = softmax(target_frame)
    _minVal, _maxVal, min_loc, max_loc = cv2.minMaxLoc(softmax_frame, None)
    print('min val of softmax', _minVal)
    print('max val of softmax', _maxVal)
    print('min loc', min_loc)
    print('max loc', max_loc)
    plt.imshow(softmax_frame)
    plt.colorbar()
    plt.show()

def test_get_softmax_frame():
    frame_num = 15

    # gets a frame from the video file
    target_frame = get_frame_np_array(VIDEO_FILE_NAME, frame_num)

    # prints a frame from the video
    show_frame_np_array(target_frame)

    #Loads template numpy array
    template_frame = np.load(TEMPLATE_FILE_NPY)

    # evals the string function for the template matching algorithm
    match_method = eval(BEST_MATCH_METHOD[0])

    # gets template match result
    result = cv2.matchTemplate(target_frame.astype(np.float32), template_frame.astype(np.float32), match_method)

    # calls get_softmax_frame
    get_softmax_frame(result)

    #prints out regular comparison
    template_match_use_all_algorithms(target_frame, TEMPLATE_FILE_PNG, 'np_array', BEST_MATCH_METHOD, frame_num)



def run_tests():
    """tests all the functions"""
    # test_frame = get_frame_np_array(VIDEO_FILE_NAME, 38)
    # show_frame_np_array(test_frame)

    # crop_frame = get_template_crop_from_frame(test_frame, 73, 251, 25, 22)
    # show_frame_np_array(crop_frame)
    # save_np_array_as_npy(crop_frame, TEMPLATE_FILE_NPY)
    # save_np_array_as_image(crop_frame, TEMPLATE_FILE_PNG)

    # matched_frame = template_match(test_frame, TEMPLATE_FILE)
    # show_frame_np_array(matched_frame)

    # result_array = get_template_match_positions(test_frame, TEMPLATE_FILE_NPY, TEMPLATE_MATCH_METHODS[0])
    # print(result_array.shape)

    # show_frame_np_array(result_array)

    # get_template_match_png(test_frame, TEMPLATE_FILE_PNG, TEMPLATE_MATCH_METHODS[0])

    # template_match_use_all_algorithms(test_frame, TEMPLATE_FILE_PNG, 'np_array', TEMPLATE_MATCH_METHODS)

    # template_match_all_frames(VIDEO_FILE_NAME)
    #write_movie()

    # data = get_max_and_max_loc_whole_file(VIDEO_FILE_NAME)
    # print(data)

    # data.plot.scatter(x='x_coord', y='y_coord')
    # plt.show()


#run_tests()
test_get_softmax_frame()
