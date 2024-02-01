"""
yolov8 Module
==================

This module provides on-device object detection capabilities by using the yolov8.
"""

from collections import deque
# import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO


import retico_core

# TODO make is so that you don't need these 3 lines below
# idealy retico-vision would be in the env so you could 
# import it by just using:
# from retico_vision.vision import ImageIU, DetectedObjectsIU
import sys
prefix = '../../'
sys.path.append(prefix+'retico-vision')

from retico_vision.vision import ImageIU, DetectedObjectsIU

class Yolov8(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Yolov8 Object Detection Module"
    
    @staticmethod
    def description():
        return "An object detection module using YOLOv8."
    
    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return DetectedObjectsIU

    
    MODEL_OPTIONS = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
    }
    
    def __init__(self, model=None, **kwargs):
        """
        Initialize the Object Detection Module
        Args:
            model (str): the name of the yolov8 model
        """
        super().__init__(**kwargs)

        if model not in self.MODEL_OPTIONS.keys():
            print("Unknown model option. Defaulting to n (yolov8n).")
            print("Other options include 's' 'm' 'l' and 'x'.")
            print("See https://docs.ultralytics.com/tasks/detect/#models for more info.")
            model = "n"
        
        self.model = YOLO(f'yolov8{model}.pt')
        self.queue = deque(maxlen=1)

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)
    
    def _detector_thread(self):
        while self._detector_thread_active:
            # time.sleep(2) # remove this
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.payload # assume PIL image

            # tic = time.time()
            results = self.model.predict(image, save=False, verbose=False)
            # toc = time.time()
            # print((toc - tic)*1000, 'ms')

            # for single image, batch size is 1
            valid_boxes = results[0].boxes.xyxy.cpu().numpy()
            # valid_score = results[0].boxes.conf.cpu().numpy()
            # valid_cls = results[0].boxes.cls.cpu().numpy()
            print(valid_boxes)

            if len(valid_boxes) == 0: continue # if nothing detected return 
            
            output_iu = self.create_iu(input_iu)
            output_iu.set_detected_objects(image, valid_boxes, "bb")
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)


    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()
    
    def shutdown(self):
        self._detector_thread_active = False
