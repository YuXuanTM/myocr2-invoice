from ultralytics import YOLOv10
from torchvision import transforms

import os
import numpy as np
from deploy.python.infer import Detector


model = YOLOv10(model=r'models/best.pt')

# 设置模型目录和输出目录
model_dir = r"models/rtdetr_r18vd_6x_coco"  # 替换为你的模型目录
output_dir = r"output"  # 替换为你的输出目录
confidence_threshold = 0.3
model2 = Detector(model_dir=model_dir,
                 device='GPU',
                 run_mode='paddle',
                 batch_size=1,
                 cpu_threads=1,
                 enable_mkldnn=True,
                 enable_mkldnn_bfloat16=True,
                 output_dir=output_dir,
                 threshold=confidence_threshold,
                 delete_shuffle_pass=False
                 )
labels = model2.pred_config.labels



def start(processed_img, flag='1'):
  if flag == '1':
    return rtdetr(processed_img)
  transform = transforms.ToTensor()
  processed_img = transform(processed_img).unsqueeze(0)
  results = model(processed_img)[0]
  boxes = results.boxes.data.tolist()
  names = results.names

  converted_detections = []
  labels = {}
  for obj in boxes:
    left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(
        obj[3])
    label = int(obj[5])
    if label in labels:
      continue
    labels[label] = label
    converted_detections.append([left, top, right, bottom, names[label], obj[4]])
  return converted_detections

def rtdetr(img):
  img = np.array(img).astype(np.uint8)

  results = model2.predict_image([img], visual=False)
  boxes = {}
  for e in results['boxes']:
    class_id, confidence, left, top, right, bottom = e
    if confidence < confidence_threshold:
      continue
    label = labels[int(class_id)]
    boxe = boxes.get(label, None)
    if boxe is None or boxe[5] < confidence:
      boxe = [left, top, right, bottom, label, confidence]
      boxes[label] = boxe
      continue

  # for box in boxes.values():
  return boxes.values()
