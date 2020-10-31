import numpy as np
import pathlib
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display


import cv2

cap = cv2.VideoCapture(1)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('../..')
#elif not pathlib.Path('models').exists():
#  get_ipython().system(u'git clone --depth 1 https://github.com/tensorflow/models')


utils_ops.tf = tf.compat.v1

tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir), None, "/home/zac/models/object_detection/data")
  model = model.signatures['serving_default']

  return model

PATH_TO_LABELS = '/home/zac/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
detection_model = load_model(model_name)


#print(detection_model.inputs)

detection_model.output_dtypes

detection_model.output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict


def show_inference(model, image_np):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  #image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.

  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']
  length = 0
  if len(classes) == len(scores):
      length = len(classes)
  return_dict = {}
  for i in range(length):
      if(scores[i] > 0.8):
        #print category_index[classes[i]]
        return_dict[i] = {category_index[classes[i]]['name']: scores[i]}

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  #display(Image.fromarray(image_np))
  cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
  return return_dict

while True:
  ret, image_np = cap.read()
  gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
  return_dict = show_inference(detection_model, image_np)
  #print return_dict
  length = len(return_dict)
  for i in range(length):
      for key in return_dict[i].keys():
          print(round((return_dict[i][key] * 100), 2), "% chance it's a ", key) 
  if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    break


#model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
#masking_model = load_model("mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28")


#masking_model.output_shapes
