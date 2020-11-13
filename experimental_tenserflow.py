#!/usr/bin/env python3
# Copyright (C) Zachary OBrien - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Zachary OBrien <zacharyaob@gmail.com>, September 2020

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import threading
import time
import helpers
import linear_assignment
import tracker
import helpers

from collections import defaultdict, deque
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

tf.debugging.set_log_device_placement(True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
print("Capturing Video")
# Define the video stream
cap = cv2.VideoCapture(1)  # Change only if you have more than one webcams
THRESHOLD = 0.8

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/zac/models/research/object_detection/data/', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90
#NUM_CLASSES = 20

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 4  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = False #True

print("defined variables")
# Download Model
if not os.path.isdir(MODEL_NAME):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
print("model is now downloaded")

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("model loaded")

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def trim_score_array(input_array):
    score_array = input_array[0]
    output_score = []
    #print("trim_score_array")
    #print(input_array)
    for x in range(len(score_array)):
        if score_array[x] > THRESHOLD:
            output_score.append(score_array[x])
    output_score_wrapper = [output_score]
    #print(output_score_wrapper)
    #print(output_score)
    #print(len(output_score))
    return output_score_wrapper, len(output_score)

def trim_array(input_array, trim_length, numpy_array=False):
    inside_array = input_array[0]
    #print(type(inside_array))
    output_array = []
    #print("TRIMMING ARRAY")
    #print("input_array: ")
    #print(input_array)
    #print("trim_length")
    #print(trim_length)
    for x in range(trim_length):
        #print("append: ")
        #print(num_array)
        #print(type(inside_array[x]))
        #print(inside_array[x])
        output_array.append(np.array(inside_array[x]))
    #print("END TRIMMING ARRAY")
    if numpy_array:
        #output_array = np.asarray(output_array)
        output_wrapper = np.array(output_array)
    else:
        output_wrapper = [output_array]
    #print("output_array")
    output1 = np.array([output_wrapper])
    #print(type(output1))
    #print(output1)
    #output2 = np.array(output_wrapper)
    #print(type(output2))
    #print(output2)
    #if numpy_array:
    #    exit()
    return output1

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    #print("ASSIGN_DETECTIONS_TO_TRACKERS")
    #print(trackers)
    #print(detections)
    #print("END ASSIGN_DETECTIONS_TO_TRACKERS")
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment.linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    

def detect ():
    global detection_graph
    global next_boxes
    global next_classes
    global next_scores
    global next_num_detections
    global current_image
    global sess
    #image_np = current_image
    #print("detect")
    #print(current_image)
    start_time = int(round(time.time() * 1000))
    image_np_expanded = np.expand_dims(current_image, axis=0)
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes
    next_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Extract detection scores
    next_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # Extract detection classes
    next_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Extract number of detections
    #print(trim_detection_array(scores, sess))
    next_num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    #print("Detect assignments finished in: ", int(round(time.time() * 1000))-start_time, "ms")
    # Actual detection.
    #(boxes, scores, classes, num_detections) = sess.run(
    #    [boxes, scores, classes, num_detections],
    #    feed_dict={image_tensor: image_np_expanded})
    temp_dict = {image_tensor: image_np_expanded}
    #print(temp_dict)
    (next_boxes, next_scores, next_classes, next_num_detections) = sess.run(
        [next_boxes, next_scores, next_classes, next_num_detections],
        feed_dict=temp_dict)
    #print(next_boxes)
    next_scores, trim_length = trim_score_array(next_scores)
    #print("after trim_score_array")
    #print(next_scores)
    #print(trim_length)
    next_boxes = trim_array(next_boxes, trim_length, True)
    next_classes = trim_array(next_classes, trim_length)
    #next_num_detections = trim_array(next_num_detections, trim_length)
    #print(next_scores)
    #print(next_boxes)
    #current_boxes = boxes
    #current_classes = classes
    #current_scores = scores
    #current_num_detections = num_detections
    #return (boxes, scores, classes, num_detections)
    #print(next_boxes)
    #print(get_img_dims())
    pipeline(next_boxes, get_img_dims()) 
    #print("Detect finished in: ", int(round(time.time() * 1000))-start_time, "ms")
    return (next_boxes, next_scores, next_classes, next_num_detections)

def capture_image():
    global next_image
    global cap
    start_time = int(round(time.time() * 1000))
    ret, next_image = cap.read()
    #current_image = image_npi
    #print("Image Capture finished in: ", int(round(time.time() * 1000))-start_time, "ms")
    return (ret, next_image)

def get_img_dims():
    global cap
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return height, width

def visualize():
    #print("visualize:")
    #print(current_boxes)
    #print(current_classes)
    #print(current_scores)
    start_time = int(round(time.time() * 1000))
    #vis_util.visualize_boxes_and_labels_on_image_array(
    #    current_image,
    #    #np.squeeze(boxes),
    #    #np.squeeze(classes).astype(np.int32),
    #    #np.squeeze(scores),
    #    np.squeeze(current_boxes),
    #    np.squeeze(current_classes).astype(np.int32),
    #    np.squeeze(current_scores),
    #    category_index,
    #    use_normalized_coordinates=True,
    #    line_thickness=8,
    #    min_score_thresh=.8)
    # Display output
    cv2.imshow('object detection', cv2.resize(current_image, (800, 600)))
    #cv2.imshow('object detection', cv2.resize(current_image, (1280, 720)))
    #print("Visualize Capture finished in: ", int(round(time.time() * 1000))-start_time, "ms")
    return

def pipeline(boxes, dims):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global next_image
    
    frame_count+=1
    
    #img_dim = (img.shape[1], img.shape[0])
    pixel_boxes = helpers.box_array_to_pixels(boxes, dims)
    z_box = pixel_boxes #det.get_localization(img) # measurement
    if debug:
       print('Frame:', frame_count)
       
    x_box =[]
    if debug: 
        for i in range(len(z_box)):
           img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
           plt.imshow(img1)
        plt.show()
    
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
    
    
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)
    
         
    # Deal with matched detections     
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            print(xx)
            tmp_trk.box =xx
            tmp_trk.hits += 1
            print("NUM HITS:")
            print(tmp_trk.hits)
            #print(tmp_trk.location_history)
            #print(np.asarray([np.array(xx)]))
            #print("Now to concatenate")
            tmp_trk.location_history = np.concatenate((tmp_trk.location_history, np.asarray([np.array(xx)])))
            #print(np.array(xx))
            print(tmp_trk.location_history)
            tmp_trk.no_losses = 0
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            print("UNMATCHED DETECTINOS")
            #print(z)
            z = np.expand_dims(z, axis=0).T
            #print(z)
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            #print("trouble x") 
            #print(x)
            #print("end trouble x") 
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            #print(track_id_list)
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)
            #print("END UNMATCHED DETECTIONS")
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
                   
       
    # The list of tracks to be annotated  
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             #img = helpers.draw_box_label(img, x_cv2) # Draw the bounding boxes on the 
             next_image = helpers.draw_box_label(next_image, x_cv2)
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    
       
    #return img
    


def test_function_call(var_to_print):
    print(var_to_print)
    return

sess = None

current_image = None
next_image = None

current_boxes = None
next_boxes = None

current_classes = None
next_classes = None

current_scores = None
next_scores = None

current_num_detections = None
next_num_detections = None

print("now looping")
# Detection
with detection_graph.as_default():
#    with tf.compat.v1.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess_tmp:
    with tf.compat.v1.Session(graph=detection_graph) as sess_tmp:
        sess = sess_tmp
        millis_prev = 0
        capture_image()
        current_image = next_image
        detect()
        current_boxes = next_boxes
        current_classes = next_classes
        current_scores = next_scores
        rent_num_detections = next_num_detections
        visualize()
        while True:
            millis = int(round(time.time() * 1000))
            print(1000/(millis-millis_prev), "FPS")
            millis_prev = millis
            # Read frame from camera
            t1 = threading.Thread(target=capture_image, args=())
            t1.start()
           
            t2 = threading.Thread(target=detect, args=())
            t2.start()
            
            t3 = threading.Thread(target=visualize, args=())
            t3.start()
            
            #t1.join()
            t2.join()
            t3.join()
            
            current_image = next_image
            current_boxes = next_boxes
            current_classes = next_classes
            current_scores = next_scores
            rent_num_detections = next_num_detections

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

