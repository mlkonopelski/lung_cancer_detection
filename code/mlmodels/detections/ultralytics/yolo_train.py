from ultralytics import YOLO

import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

# Load a model
model = YOLO("yolov8n.yaml", verbose=True)  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

model.train(cfg = 'code/mlmodels/detections/ultralytics/training.yaml')
metrics = model.val()  # evaluate model performance on the validation set
results = model("code/mlmodels/detections/ultralytics/.data/images/train/1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208.png")  # predict on an image

# path = model.export(format="onnx")  # export the model to ONNX format