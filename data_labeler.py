import json
import uuid

import cv2
import os
import JsonToXML
import xmltodict
import xml.etree.cElementTree as ET
import xml.dom.minidom
from PIL import Image, ImageDraw
import numpy as np


class data_labeler():
    def __init__(self, image=None, boxes=[], object_name="output", object_dir="~/"):
        self.image = image
        self.boxes = boxes
        self.object_name = object_name
        self.object_dir = object_dir

    def load_voc(self, img_file, xml_file):
        with open(xml_file) as voc_file:
            data_dict = xmltodict.parse(voc_file.read())
            self.full_path = data_dict["path"]
            self.object_name = self.full_path.split("/")[-1]
            self.full_path = self.full_path.replace(self.object_name, "")
            temp_boxes = []
            for object in data_dict["object"]:
                single_box = [object["xmin"], object["ymin"],object["xmax"],object["ymax"]]
                temp_boxes.append(single_box)
            self.boxes = temp_boxes

            pil_img = Image.open(img_file)
            draw = ImageDraw.Draw(pil_img)
            for box in self.boxes:
                draw.rectangle(box, outline='red')

            open_cv_image = np.array(pil_img)
            detection_id = str(uuid.uuid1())
            cv2.imshow("Confirm image", open_cv_image)

            while(True):
                key = cv2.waitKey(0)
                # Press "q" to quit
                if 0xFF == ord("q"):
                    print(key)
                    cv2.destroyAllWindows()
                    exit()
                elif 0xFF == ord("m"):
                    print(key)
                    #cv2.imwrite("output/manual_saves/" + detection_id + ".png", save_image)
                elif key:
                    print(key)

        xml_file.close()

    def save_voc(self):
        json_object_list = dict()
        json_object_list["filename"] = self.object_name +".png"
        json_object_list["path"] = self.object_dir
        #TODO Folder, Path, Source

        width, height, depth = self.image.shape
        size = {"width": width, "height": height, "depth": depth}
        json_object_list["size"] = size

        segmented = {"segmented":"0"}
        json_object_list["segmented"] = "0"

        json_object_list["object"] = []
        for bounding_box in self.boxes:
            #name = bounding_box["label"]
            name = "person"
            pose = "Unspecified"
            trunkated = "0"
            difficult = "0"
            xmin = bounding_box[0]
            ymin = bounding_box[1]
            xmax = bounding_box[2]
            ymax = bounding_box[3]
            bndbox = {"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax}
            my_object = {"name":name, "pose":pose, "trunkated":trunkated, "difficult":difficult, "bndbox":bndbox}
            json_object_list["object"].append(my_object)

        root = JsonToXML.fromText(json.dumps(json_object_list),
                                  rootName="annotation")  # convert the file to XML and return the root node
        xmlData = ET.tostring(root, encoding='utf8', method='xml').decode()  # convert the XML data to string
        dom = xml.dom.minidom.parseString(xmlData)
        prettyXmlData = dom.toprettyxml()  # properly format the string of XML data
        #print(prettyXmlData)  # print the formatted XML data

        png_filename = self.object_dir + self.object_name + ".png"
        xml_filename = self.object_dir + self.object_name + ".xml"

        cv2.imwrite(png_filename, self.image)
        # img = Image.fromarray(np.uint8(self.image)).convert('RGB')
        # img.save(png_filename)

        with open(xml_filename, "w+") as output_file:
            output_file.write(prettyXmlData)



def load_voc(img_file, xml_file):
    with open(xml_file) as voc_file:
        data_dict = xmltodict.parse(voc_file.read())
        data_dict = json.dumps(data_dict)
        data_dict = json.loads(data_dict)['annotation']
        print(data_dict)
        full_path = data_dict["path"]
        object_name = full_path.split("/")[-1]
        full_path = full_path.replace(object_name, "")
        temp_boxes = []
        for object in data_dict["object"]:
            object = object['bndbox']
            single_box = [object["xmin"], object["ymin"],object["xmax"],object["ymax"]]
            temp_boxes.append(single_box)
        boxes = temp_boxes

        pil_img = Image.open(img_file)
        draw = ImageDraw.Draw(pil_img)
        for box in boxes:
            p1 = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]
            draw.rectangle(p1, outline='red')

        open_cv_image = np.array(pil_img)
        detection_id = str(uuid.uuid1())
        cv2.imshow("Confirm image", open_cv_image)

        while(True):
            key = cv2.waitKey(0)
            # Press "q" to quit
            if key == ord('w'):
                #119 = w
                #115 = s
                print("w")
                print(key)
                #cv2.imwrite("output/manual_saves/" + detection_id + ".png", save_image)
            elif key == ord("s"):
                #119 = w
                #115 = s
                print("s")
                print(key)
                #cv2.imwrite("output/manual_saves/" + detection_id + ".png", save_image)
            elif key:
                print("Unknown Key")
                print(key)
                cv2.destroyAllWindows()
                exit()

    xml_file.close()

if __name__ == '__main__':
    load_voc("output/new_training_images/2fa8cadc-2217-11eb-b112-99ac4dcb8e94.png","output/new_training_images/2fa8cadc-2217-11eb-b112-99ac4dcb8e94.xml")