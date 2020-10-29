import cv2
import os
from collections import defaultdict, deque
import urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
import numpy as np
import linear_assignment
import tracker
import helpers
import time
import keras
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
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
PATH_TO_LABELS = os.path.join('/home/zac/models/research/object_detection/data/', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

max_age = 4  # no.of consecutive unmatched detection before
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = False #True

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

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    # (im_width, im_height) = image.size
    # return np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)
    return np.array(Image.open(image))


def trim_score_array(input_array):
    score_array = input_array[0]
    output_score = []
    # print("trim_score_array")
    # print(input_array)
    for x in range(len(score_array)):
        if score_array[x] > THRESHOLD:
            output_score.append(score_array[x])
    output_score_wrapper = [output_score]
    # print(output_score_wrapper)
    # print(output_score)
    # print(len(output_score))
    return output_score_wrapper, len(output_score)


def trim_array(input_array, trim_length, numpy_array=False):
    inside_array = input_array[0]
    # print(type(inside_array))
    output_array = []
    # print("TRIMMING ARRAY")
    # print("input_array: ")
    # print(input_array)
    # print("trim_length")
    # print(trim_length)
    for x in range(trim_length):
        # print("append: ")
        # print(num_array)
        # print(type(inside_array[x]))
        # print(inside_array[x])
        output_array.append(np.array(inside_array[x]))
    # print("END TRIMMING ARRAY")
    if numpy_array:
        # output_array = np.asarray(output_array)
        output_wrapper = np.array(output_array)
    else:
        output_wrapper = [output_array]
    # print("output_array")
    output1 = np.array([output_wrapper])
    # print(type(output1))
    # print(output1)
    # output2 = np.array(output_wrapper)
    # print(type(output2))
    # print(output2)
    # if numpy_array:
    #    exit()
    return output1


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''
    # print("ASSIGN_DETECTIONS_TO_TRACKERS")
    # print(trackers)
    # print(detections)
    # print("END ASSIGN_DETECTIONS_TO_TRACKERS")
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

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

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

IMAGE_FOLDER = "ball_images/"

IMAGE_PATHS = os.listdir(IMAGE_FOLDER)
IMAGE_PATHS = [IMAGE_FOLDER + sub for sub in IMAGE_PATHS]
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
#for image_path in IMAGE_PATHS:
def detect(image_path):
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

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
    #print(detections['detection_boxes'])
    print(detections['detection_scores'])
    plt.figure()
    plt.imsave('output/' + str('detections') + '.jpg', image_np_with_detections)

    # plt.figure()
    # plt.imshow(image_np_with_detections)
    print('Done')
#plt.show()







# for image_path in IMAGE_PATHS:
#     image_np = np.array(Image.open(file_name))
#     input_tensor = tf.convert_to_tensor(image_np)
#
#     input_tensor2 = input_tensor[tf.newaxis, ...]
#     print("Model 1 with input_tensor2")
#
#     detections = new_model1(input_tensor2)
#     print("Predictions complete")
#     num_detections = int(detections.pop('num_detections'))
#
#     detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
#
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
#
#     image_np_with_detections = image_np.copy()
#
#     # detections['num_detections'] = num_detections
#     #
#     # for key, value in detections.items():
#     #     detections[key] = value[0]
#     #
#     # image_np_with_detections = cv2.imread(file_name)
#     # (h, w) = image_np_with_detections.shape[:2]
#
#     viz_utils.visualize_boxes_and_labels_on_image_array(
#         image_np_with_detections,
#         detections['detection_boxes'],
#         detections['detection_classes'],
#         detections['detection_scores'],
#         category_index,
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=200,
#         min_score_thresh=.30,
#         agnostic_mode=False)
#
#
#     plt.figure()
#     plt.imshow(image_np_with_detections)
#     print('Done')
# plt.show()

# for index, box in enumerate(detections['detection_boxes']):
#     score = detections['detection_scores'][index].numpy()
#     if score > THRESHOLD:
#         startX = int(detections['detection_boxes'][index][0] * w)
#         startY = int(detections['detection_boxes'][index][1] * h)
#         endX = int(detections['detection_boxes'][index][2] * w)
#         endY = int(detections['detection_boxes'][index][3] * h)
#         print("Start x")
#         print(startX)
#         print("startY")
#         print(startY)
#         print("endX")
#         print(endX)
#         print("endY")
#         print(endY)
#         y = startY - 10 if startY - 10 > 10 else startY + 10
#         detection_class = str(detections['detection_classes'][index].numpy())
#         cv2.putText(image_np_with_detections, detection_class, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
#         cv2.rectangle(image_np_with_detections, (startX, startY), (endX, endY), (0, 255, 0), 2)

# cv2.imshow("Output", image_np_with_detections)
# cv2.waitKey(0)



#Detections object has
    # proposal_boxes_normalized
    # detection_anchor_indices
    # raw_detection_boxes
    # class_predictions_with_background
    # box_classifier_features
    # proposal_boxes
    # rpn_features_to_crop
    # rpn_objectness_predictions_with_background
    # mask_predictions
    # detection_boxes
    # detection_masks
    # refined_box_encodings
    # final_anchors
    # rpn_box_predictor_features
    # raw_detection_scores
    # detection_classes
    # rpn_box_encodings
    # num_proposals
    # detection_multiclass_scores
    # image_shape
    # anchors
    # detection_scores