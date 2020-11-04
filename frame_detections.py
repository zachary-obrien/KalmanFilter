#!/usr/bin/env python3
# Copyright (C) Zachary OBrien - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Zachary OBrien <zacharyaob@gmail.com>, September 2020

import numpy as np


class FrameDetection():
    def __init__(self, image, detection_boxes, detection_classes, detection_scores, default_threshold=0.7):
        self.__image = image
        self.__detection_boxes = detection_boxes
        self.__detection_classes = detection_classes
        self.__detection_scores = detection_scores
        self.__THRESHOLD = default_threshold

    def __repr__(self):
        output = "Detection Boxes Shape\n"
        output = output + ' '.join(map(str, self.__detection_boxes.shape)) + "\n"
        output = output + "Detection Classes Shape\n"
        output = output + ' '.join(map(str, self.__detection_classes.shape)) + "\n"
        output = output + "Detection Scores Shape\n"
        output = output + ' '.join(map(str, self.__detection_scores.shape)) + "\n"
        return output

    def __str__(self):
        output = "Detection Boxes\n"
        output = output + np.array_str(self.__detection_boxes) + "\n"
        output = "Detection Classes\n"
        output = output + np.array_str(self.__detection_classes) + "\n"
        output = "Detection Scores\n"
        output = output + np.array_str(self.__detection_scores) + "\n"

        return output

    def trim_by_score_threshold(self, input_threshold):
        np.savetxt('classes_array.npy', self.__detection_classes, delimiter=',')
        np.savetxt('boxes_array.npy', self.__detection_boxes, delimiter=',')
        np.savetxt('scores_array.npy', self.__detection_scores, delimiter=',')

        self.__detection_scores, length = self.trim_score_array(self.__detection_scores)
        self.__detection_classes = self.trim_array(self.__detection_classes, length, True)
        self.__detection_boxes = self.trim_array(self.__detection_boxes, length, True)

    def trim_score_array(self, input_array, input_threshold=None):
        if not input_threshold:
            input_threshold = self.__THRESHOLD
        cut_val = 0

        for index, score in enumerate(input_array):
            if score < input_threshold:
                cut_val = index
                break

        output_score = input_array[:cut_val]
        # for x in range(len(input_array)):
        #     if input_array[x] > input_threshold:
        #         output_score.append(input_array[x])
        # output_score_wrapper = [output_score]
        # return output_score_wrapper, len(output_score)
        return output_score, len(output_score)

    def trim_array(self, input_array, trim_length, numpy_array=True):
        return input_array[:trim_length]
        # output_array = []
        # for x in range(trim_length):
        #     output_array.append(np.array(input_array[x]))
        # if numpy_array:
        #     output_wrapper = np.array(output_array)
        # else:
        #     output_wrapper = [output_array]
        # output1 = np.array([output_wrapper])
        # return output1

    def get_boxes(self):
        return self.__detection_boxes

    def get_classes(self):
        return self.__detection_classes

    def get_scores(self):
        return self.__detection_scores

    def get_image(self):
        return self.__image


if __name__ == "__main__":
    # execute only if run as a script
    np_classes = np.loadtxt("classes_array.npy", delimiter=',')
    np_boxes = np.loadtxt("boxes_array.npy", delimiter=',')
    np_scores = np.loadtxt("scores_array.npy", delimiter=',')

    myFrame = FrameDetection(None, np_boxes, np_classes, np_scores)

    print("************************************")
    print("Detection")
    print(repr(myFrame))
    print("************************************")
    print("Trimming")
    myFrame.trim_by_score_threshold(0.6)
    print("************************************")
    print(repr(myFrame))
    print("************************************")
