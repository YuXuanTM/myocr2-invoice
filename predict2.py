import numpy as np
from deploy.python.infer import Detector

# CPU OR GPU
device = 'GPU'

# 设置模型目录和输出目录
model_dir = r"models/rtdetrv2"  # 替换为你的模型目录
output_dir = r"output"  # 替换为你的输出目录
confidence_threshold = 0.3
model2 = Detector(model_dir=model_dir,
                 device=device,
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



def start(processed_img):
  img = np.array(processed_img).astype(np.uint8)
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
  return boxes.values()
