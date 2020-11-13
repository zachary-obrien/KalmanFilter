# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""A demo for object detection.
For Raspberry Pi, you need to install 'feh' as image viewer:
sudo apt-get install feh
Example (Running under python-tflite-source/edgetpu directory):
  - Under the parent directory python-tflite-source.
  - Face detection:
    python3.5 edgetpu/demo/object_detection.py \
    --model='test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite' \
    --input='test_data/face.jpg'
  - Pet detection:
    python3.5 edgetpu/demo/object_detection.py \
    --model='test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite' \
    --label='test_data/pet_labels.txt' \
    --input='test_data/pets.jpg'
'--output' is an optional flag to specify file name of output image.
"""
import time
import platform
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
import mss
import cv2
import numpy as np
import data_labeler
import uuid


# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def run_inference():
    global fps, start_time, display_time
    #engine = DetectionEngine("models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
    engine = DetectionEngine("/content/tflite2/eft2.tflite")
    labels = ReadLabelFile("models/coco_labels.txt")
    monitor = {"top": 140, "left": 2580, "width": 512, "height": 512}
    sct = mss.mss()
    max_saves = 500
    current_saves = 0
    print_fps = "TBD"
    while(True):
        img = np.array(sct.grab(monitor))
        # # to ger real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        pil_img = Image.fromarray(np.uint8(img)).convert('RGB')
        save_image = img
        draw = ImageDraw.Draw(pil_img)
        ans = engine.detect_with_image(pil_img, threshold=0.5, keep_aspect_ratio=True,
                                       relative_coord=False, top_k=10)
        if ans:
            if current_saves > max_saves:
                exit()
            detection_id = str(uuid.uuid1())
            bounding_boxes = []
            found_person = False
            for obj in ans:
                if labels and obj.label_id == 0:
                    found_person = True
                    current_saves = current_saves + 1
                    print ('-----------------------------------------')
                    print(labels[obj.label_id], ' Score = ', obj.score)
                    box = obj.bounding_box.flatten().tolist()
                    bounding_boxes.append(box)

                    #print ('box = ', box)
                    # Draw a rectangle.
                    draw.rectangle(box, outline='red')

                    #img.save(output_name)
            if found_person:
                data = data_labeler.data_labeler(save_image, bounding_boxes, detection_id, "output/new_training_images/")
                data.save_voc()
                time.sleep(1)
            #exit()

        open_cv_image = np.array(pil_img)
        cv2.imshow(title, open_cv_image)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(25) & 0xFF == ord("m"):
            print("Manual Capture")
            time.sleep(1)
            cv2.imwrite("output/manual_saves/" + detection_id + ".png", save_image)

        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time:
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()


        else:
            continue
            #print ('No object detected!')

if __name__ == '__main__':
    run_inference()
# def main():
#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       '--model', help='Path of the detection model.', required=True)
#   parser.add_argument(
#       '--label', help='Path of the labels file.')
#   parser.add_argument(
#       '--input', help='File path of the input image.', required=True)
#   parser.add_argument(
#       '--output', help='File path of the output image.')
#   args = parser.parse_args()
#   if not args.output:
#     output_name = 'object_detection_result.jpg'
#   else:
#     output_name = args.output
#   # Initialize engine.
#   engine = DetectionEngine(args.model)
#   labels = ReadLabelFile(args.label) if args.label else None
#   # Open image.
#   img = Image.open(args.input)
#   draw = ImageDraw.Draw(img)
#   # Run inference.
#   ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,
#                                relative_coord=False, top_k=10)
#   # Display result.
#   if ans:
#     for obj in ans:
#       print ('-----------------------------------------')
#       if labels:
#         print(labels[obj.label_id])
#       print ('score = ', obj.score)
#       box = obj.bounding_box.flatten().tolist()
#       print ('box = ', box)
#       # Draw a rectangle.
#       draw.rectangle(box, outline='red')
#     img.save(output_name)
#     if platform.machine() == 'x86_64':
#       # For gLinux, simply show the image.
#       img.show()
#     elif platform.machine() == 'armv7l':
#       # For Raspberry Pi, you need to install 'feh' to display image.
#       subprocess.Popen(['feh', output_name])
#     else:
#       print ('Please check ', output_name)
#   else:
#     print ('No object detected!')