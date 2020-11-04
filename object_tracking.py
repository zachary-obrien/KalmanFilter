#!/usr/bin/env python3
# Copyright (C) Zachary OBrien - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Zachary OBrien <zacharyaob@gmail.com>, September 2020

import cv2
import os
from collections import defaultdict, deque
import urllib as urllib
import tarfile
import tensorflow as tf
import numpy as np
import linear_assignment
import tracker
import helpers
import template_matching
import time
import keras
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt
from frame_detections import FrameDetection

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
# cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
# cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
THRESHOLD = 0.8

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz
# MODEL_NAME = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('/home/zac/models/research/object_detection/data/', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('C:/Users/Cooper/Documents/Tensorflow/models/research/object_detection/data',
                              'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

max_age = 4  # no.of consecutive unmatched detection before
# a track is deleted

min_hits = 1  # no. of consecutive matches needed to establish a track

# reads the .png file into a cv2 image
TEMPLATE_IMAGE = cv2.imread(template_matching.TEMPLATE_FILE_PNG, cv2.IMREAD_COLOR)

frame_count = 0

tracker_list = []  # list for trackers
# list for track ID
track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = False
# debug =  True

print("defined variables")
# Download Model
model_location = MODEL_NAME + "/saved_model"
if not os.path.isdir(MODEL_NAME):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    if MODEL_FILE.endswith("tar.gz"):
        tar = tarfile.open(MODEL_FILE, "r:gz")
        tar.extractall()
        tar.close()
print("model is now downloaded")

PATH_TO_CFG = MODEL_NAME + "/pipeline.config"
PATH_TO_CKPT = MODEL_NAME + "/checkpoint"

print('Loading model... ', end='')
start_time = time.time()

detect_fn = keras.models.load_model(model_location)
print(type(detect_fn))
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

IMAGE_FOLDER = "ball_images/"

IMAGE_PATHS = os.listdir(IMAGE_FOLDER)
IMAGE_PATHS = [IMAGE_FOLDER + sub for sub in IMAGE_PATHS]
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image, from_file=True):
    # (im_width, im_height) = image.size
    # return np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)
    if from_file:
        image = cv2.imread(image)
        # Image.open(image)
    return np.array(image)


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3, current_frame=0):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    '''
    # print("ASSIGN_DETECTIONS_TO_TRACKERS")
    # print(trackers)
    # print(detections)
    # print("END ASSIGN_DETECTIONS_TO_TRACKERS")

    unmatched_trackers, unmatched_detections = [], []
    matches = []

    if current_frame == 1:
        print("First Frame")
        matches = np.empty((0, 2), dtype=int)
        for d, det in enumerate(detections):
            unmatched_detections.append(d)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        # trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = helpers.box_iou2(trk, det)

            # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment.linear_assignment(-IOU_mat)

    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# def detect (current_image):
#     global new_model
#     start_time = int(round(time.time() * 1000))
#
#     img_array = np.array(Image.open(current_image))
#
#     #image_np_expanded = np.expand_dims(current_image, axis=0)
#     input_tensor = tf.convert_to_tensor(img_array)
#     input_tensor = input_tensor[tf.newaxis, ...]
#     detections = new_model.predict(input_tensor)
#     print(detections)

# for image_path in IMAGE_PATHS:
def detect(image, from_file=True):
    # print('Running inference for {}... '.format(image), end='')

    image_np = load_image_into_numpy_array(image, from_file)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # Template match video frame with template image to get cv2.TM_CCORR_NORMED value matrix
    result = template_matching.get_template_match_positions(image, template_matching.TEMPLATE_FILE_NPY,
                                                            template_matching.BEST_MATCH_METHOD[0])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result, None)
    print('max value and coord location: ', max_val, max_loc)

    # determine value for update f input TODO

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.70,
        agnostic_mode=False)

    # print(detections['detection_boxes'])
    # print(detections['detection_scores'])
    # print(type(detections['detection_classes']))
    # plt.figure()
    # plt.imsave('output/' + str('detections') + '.jpg', image_np_with_detections)
    # plt.show()
    # plt.figure()
    # plt.imshow(image_np_with_detections)
    print('Done')
    frame_detect_object = FrameDetection(image_np_with_detections, detections['detection_boxes'],
                                         detections['detection_classes'], detections['detection_scores'])
    frame_detect_object.set_max_val_and_loc(max_val, max_loc)

    return frame_detect_object

    # add process covariance tuple to return tuple


def pipeline(boxes, image, max_val, max_loc):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug

    # sets template match box dimensions based on size of template image
    template_bb_x, template_bb_y = max_loc[0] + TEMPLATE_IMAGE.shape[0], max_loc[1] + TEMPLATE_IMAGE.shape[1]

    # draws boxes on image
    cv2.rectangle(image, max_loc, (template_bb_x, template_bb_y), (0, 255, 0), 1, 8, 0)

    frame_count += 1

    img_dim = (image.shape[1], image.shape[0])
    pixel_boxes = helpers.box_array_to_pixels(boxes, img_dim)
    z_box = pixel_boxes  # det.get_localization(img) # measurement
    # z_box = boxes
    img = image
    if debug:
        print('Frame:', frame_count)

    x_box = []
    # if debug:
    #     for i in range(len(z_box)):
    #         img1 = helpers.draw_box_label(image, z_box[i], box_color=(255, 0, 0))
    #         plt.imshow(img1)
    #     plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3, current_frame=frame_count)

    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)

    # Deal with matched detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            print(xx)
            tmp_trk.box = xx
            tmp_trk.hits += 1
            print("NUM HITS:")
            print(tmp_trk.hits)
            # print(tmp_trk.location_history)
            # print(np.asarray([np.array(xx)]))
            # print("Now to concatenate")
            tmp_trk.location_history = np.concatenate((tmp_trk.location_history, np.asarray([np.array(xx)])))
            # print(np.array(xx))
            print(tmp_trk.location_history)
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            print("UNMATCHED DETECTIONS")
            # print(z)
            z = np.expand_dims(z, axis=0).T
            # print(z)
            tmp_trk = tracker.Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            # print("trouble x")
            # print(x)
            # print("end trouble x")
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            print(track_id_list)
            tmp_trk.id = track_id_list.popleft()  # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)
            print("END UNMATCHED DETECTIONS")

    # Deal with unmatched tracks
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # The list of tracks to be annotated
    good_tracker_list = []
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            if debug:
                print('updated box: ', x_cv2)
                print()
            # img = helpers.draw_box_label(img, x_cv2) # Draw the bounding boxes on the
            # img = helpers.draw_box_label(image, x_cv2)
            # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    return img


def run_flow():
    cap = cv2.VideoCapture('singleball.mov')
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            print('frame: ', frame_number)
            frame_number += 1
            frame_detection = detect(frame, from_file=False)
            frame_detection.trim_by_score_threshold(0.8)
            labeled_output = pipeline(frame_detection.get_boxes(), frame_detection.get_image(),
                                      frame_detection.get_max_value(), frame_detection.get_max_location())
            cv2.imshow('frame', labeled_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute only if run as a script
    run_flow()
