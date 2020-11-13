
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import random
import glob
import mss
import cv2
import time

#%matplotlib inline
matplotlib.use( 'tkagg' )

class tf_detector:
    def __init__(self):
        self.monitor = {"top": 140, "left": 2580, "width": 512, "height": 512}
        pipeline_config = "/content/fine_tuned_model/pipeline.config"
        # generally you want to put the last ckpt from training in here
        # model_dir = '/content/training'
        model_dir = '/content/fine_tuned_model/checkpoint/ckpt-0'
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(
            model=detection_model)
        ckpt.restore(model_dir)
        # ckpt.restore(os.path.join('/content/training/'))

        self.detect_fn = self.get_model_detection_function(detection_model)

        label_map_path = configs['eval_input_config'].label_map_path
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    def get_model_detection_function(self, model):
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

    # Function to read labels from text files.
    def ReadLabelFile(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def go_detect(self):

        # title of our window
        title = "FPS benchmark"
        # set start time to current time
        start_time = time.time()
        # displays the frame rate every 2 second
        display_time = 2
        # Set primarry FPS to 0
        fps = 0


        sct = mss.mss()
        while (True):
            img = np.array(sct.grab(self.monitor))
            # # to ger real color we do this:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            pil_img = Image.fromarray(np.uint8(img)).convert('RGB')
            draw = ImageDraw.Draw(pil_img)
            image_np = self.load_image_into_numpy_array(pil_img)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)

            image_np_with_detections = image_np.copy()

            detections, predictions_dict, shapes = self.detect_fn(input_tensor)

            # elif cv2.waitKey(25) & 0xFF == ord("m"):
            #     print("Manual Capture")
            #     time.sleep(1)
            #     cv2.imwrite("output/manual_saves/" + detection_id + ".png", save_image)

            label_id_offset = 1
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=2000,
                min_score_thresh=.8,
                agnostic_mode=False,
                skip_scores=False,
                skip_labels=True,
                line_thickness=1,
            )
            open_cv_image = np.array(image_np_with_detections)
            cv2.imshow("title", open_cv_image)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

            fps += 1
            TIME = time.time() - start_time
            if (TIME) >= display_time:
                print("FPS: ", fps / (TIME))
                fps = 0
                start_time = time.time()

    def load_image_into_numpy_array(self, image, is_file=False):
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


# #This is where the images get input. Lets push this into a fxn
# TEST_IMAGE_PATHS = glob.glob('/content/test/test/*.jpg')
# image_path = random.choice(TEST_IMAGE_PATHS)
#
# image_np = my_detector.load_image_into_numpy_array(image_path)
my_detector = tf_detector()
my_detector.go_detect()
