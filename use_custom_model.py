import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import time
import numpy as np
import cv2
import mss
import glob


import random

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  print(im_width, im_height)
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    print(shapes)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn







#recover our saved model
pipeline_config = "/content/fine_tuned_model/pipeline.config"
#generally you want to put the last ckpt from training in here
# model_dir = '/content/training'
model_dir = '/content/fine_tuned_model/checkpoint/ckpt-0'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

detect_fn = get_model_detection_function(detection_model)
print("Created Detector")



TEST_IMAGE_PATHS = glob.glob('/content/test/test/*.jpg')
image_path = random.choice(TEST_IMAGE_PATHS)
image_np = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)

detections, predictions_dict, shapes = detect_fn(input_tensor)

print(detections['detection_scores'])
label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=2000,
      min_score_thresh=.9,
      agnostic_mode=False,
      skip_scores=False,
      skip_labels=True,
      line_thickness=1,
)


plt.figure(figsize=(12,16))
plt.imshow(image_np_with_detections)
plt.show()


















# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(model_dir)
# ckpt.restore(os.path.join('/content/training/'))





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

        frame_detection = detect_fn(img)
        print(frame_detection)


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

