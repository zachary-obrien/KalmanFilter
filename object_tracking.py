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
MODEL_NAME = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'
#MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
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

new_model1 = keras.models.load_model(model_location)
print(type(new_model1))
new_model2= tf.saved_model.load(model_location)
print(type(new_model2))
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
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


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


def detect (current_image):
    global new_model
    start_time = int(round(time.time() * 1000))

    img_array = np.array(Image.open(current_image))

    #image_np_expanded = np.expand_dims(current_image, axis=0)
    input_tensor = tf.convert_to_tensor(img_array)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = new_model.predict(input_tensor)
    print(detections)

file_name = "person_test_images/Lake-Calhoun_MAIN_2.jpg"

img2 = np.array(Image.open(file_name))
input_tensor = tf.convert_to_tensor(img2)

input_tensor2 = input_tensor[tf.newaxis, ...]
print("Model 1 with input_tensor2")
try:
    detections = new_model1(input_tensor2)
    print("detections")
    print(detections)
    print(detections.items())
    num_detections = int(detections.pop('num_detections'))
    print("num_detections")
    print(num_detections)
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


    # print(detections.items())
    for key, value in detections.items():
        print(key)
        detections[key] = value[0]

    image_np_with_detections = cv2.imread(file_name)
    (h, w) = image_np_with_detections.shape[:2]
    for index, box in enumerate(detections['detection_boxes']):
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image_np_with_detections, detections['detection_classes'][index], (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
        cv2.rectangle(image_np_with_detections, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.imshow("Output", image_np_with_detections)
        cv2.waitKey(0)

except Exception as e:
    print("ERROR")
    print(e)



