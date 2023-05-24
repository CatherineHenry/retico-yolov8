# retico-yolov8
A ReTiCo module for yolov8. See below for more information on the models.

### Installation and requirements

### Example
```python
import sys
from retico import *

prefix = '/path/to/module/'
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-yolov8')

# from retico_yolov8 import YoloV8
from retico_vision.vision import WebcamModule 
from retico_yolov8.yolov8 import Yolov8



webcam = WebcamModule()
yolo = Yolov8()
debug = modules.DebugModule()

webcam.subscribe(yolo)
yolo.subscribe(debug)

run(webcam)

print("Network is running")
input()

stop(webcam)
```

[More information on models from the yolov8 github repository](https://github.com/ultralytics/ultralytics)

Citation
```
```