"""
yolov8 Module
==================

This module provides on-device object detection capabilities by using the yolov8.
"""

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import retico_core
from PIL import Image
from retico_vision.vision import ImageIU, DetectedObjectsIU
from ultralytics import YOLO
from ultralytics.utils.ops import clip_boxes
from ultralytics.utils.plotting import Annotator


# TODO make is so that you don't need these 3 lines below
# ideally retico-vision would be in the env so you could
# import it by just using:
# from retico_vision.vision import ImageIU, DetectedObjectsIU
# prefix = '../../'
# sys.path.append(prefix+'retico-vision')

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
        self.base_filepath = './yolo_output'

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

            # clips the bounding boxes to the image shape
            clipped_boxes = clip_boxes(valid_boxes, results[0].orig_shape)

            # Save the input image with the bounding box (if present)
            path = Path(f"{self.base_filepath}/{input_iu.execution_uuid}/undetected/")
            # Annotate a copy of the image, otherwise it will annotate the original image.
            # The bounding box border will show in the sub image if the annotated image is used for clipping
            annotated_image = np.asarray(image)
            if len(valid_boxes) > 0:
                annotator = Annotator(annotated_image)
                for box in valid_boxes:
                    print(box)
                    annotator.box_label(box, 'n/a')
                path = Path(f"{self.base_filepath}/{input_iu.execution_uuid}/detected/")
                annotated_image = annotator.result()
            path.mkdir(parents=True, exist_ok=True)
            file_name = f"{input_iu.flow_uuid}.png" # TODO: png or jpg better?
            imwrite_path = f"{str(path)}/{file_name}"
            try:
                Image.fromarray(annotated_image).save(imwrite_path)
            except FileNotFoundError:
                print(f"Did not write YOLO output to {imwrite_path}. Check directory exists.")


            output_iu = self.create_iu(input_iu)
            if len(clipped_boxes) == 0:
                output_iu.set_detected_objects(image, [], "bb")
            else:
                output_iu.set_detected_objects(image, clipped_boxes, "bb")
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)


    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()
    
    def shutdown(self):
        self._detector_thread_active = False
