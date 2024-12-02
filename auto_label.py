import cv2
import os
import json
import numpy as np

from main import img_joint
from ultralytics import YOLOv10
from PIL import Image
from main import preprocess_image
from main import convert_coordinates
from main import __get_img__

model = YOLOv10(model=r'models/best.pt')

FIXED_WIDTH = 1219

# 自动标注
# 使用PPOCRLabel打开target_path路径进行微调
def auto_label():
  # 待标注的目标路径
  directory = "D:\发票\发票3"
  # directory = r"D:\发票\fapiao"
  files = os.listdir(directory)
  # 标注信息保存路径.
  target_path = 'img2'
  # 缩放与原图可视化图片保存路径
  contrast_path = 'flag'
  # 文件名字, 自增.
  count = 1200
  file = open(target_path + "/Label.txt", 'w', encoding='utf-8')
  for file_name in files:
    count = count + 1
    file_path = os.path.join(directory, file_name)
    img = __get_img__(file_name, file_path)
    processed_img, orig_size, new_size, paste_coords, canvas = preprocess_image(img)
    img_file = target_path + "/" + f"{count}" +".png"
    # 保存图片, 缩放后的.
    canvas.save(img_file)
    # cv2.imwrite(imgfile, canvas)

    results = model(processed_img)[0]
    boxes = results.boxes.data.tolist()
    names = results.names
    data_line = []
    labels = {}
    canvas = np.array(canvas)
    for obj in boxes:
      left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
      label = int(obj[5])
      confidence = obj[4]
      if label in labels:
        continue
      labels[label] = label
      cv2.rectangle(canvas, (left, top), (right, bottom), color=(127, 255, 255), thickness=2, lineType=cv2.LINE_AA)
      caption = f"{label} {confidence:.2f}"
      w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
      cv2.rectangle(canvas, (left - 3, top - 33), (left + w + 10, top), (127, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
      cv2.putText(canvas, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
      box = convert_coordinates([left, top, right, bottom], orig_size, new_size, paste_coords)
      o_left, o_top, o_right, o_bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      img = np.array(img)
      cv2.rectangle(img, (o_left, o_top), (o_right, o_bottom),color=(127, 255, 255), thickness=2, lineType=cv2.LINE_AA)
      points = [[left, top], [right, top], [right, bottom], [left, bottom]]
      data = {
        "transcription": names[label],
        "points": points,
        "difficult": False
      }
      data_line.append(data)
    json_data = json.dumps(data_line)
    # 保存对比图.
    img_joint(Image.fromarray(img), Image.fromarray(canvas), 1).save(contrast_path + "/" + f"{count}" + ".png")
    aa =img_file + "	" + json_data
    print(img_file)
    # 写入标注位置信息.
    file.write(aa + '\n')

auto_label()