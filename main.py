import cv2
import numpy as np
from ultralytics import YOLOv10
from paddleocr import PaddleOCR
from flask import Flask, request
from PIL import Image

app = Flask(__name__)

ocr = PaddleOCR(
    rec=r'models/ch_PP-OCRv4_rec_infer',
    det=r'models/ch_PP-OCRv4_det_infer',
    use_gpu=True)
model = YOLOv10(model=r'models/best.pt')
types = ['image/png', 'image/jpg', 'image/jpeg']

@app.route('/invoice_ocr', methods=['POST'])
def invoice_ocr():
  uploaded_file = request.files['file']
  print(uploaded_file.content_type)
  if uploaded_file is None or uploaded_file.content_type not in types:
    return {}
  img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
  results = model(img)[0]
  boxes = results.boxes.data.tolist()
  names = results.names
  ocrResult = {}
  for obj in boxes:
    left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
    label = int(obj[5])
    cropped_img = img[top:bottom, left:right]
    rr = ocr.ocr(cropped_img, det=False, cls=False)
    for line in rr:
      if line is None:
        continue
      for word_info in line:
        print(names[label], word_info[0])
        ocrResult[names[label]] = word_info[0]
  return ocrResult

def img_joint(new_img, old_img, axis=0):
  w1, h1 = old_img.size
  w2, h2 = new_img.size
  max_width = max(w1, w2) + 4
  max_height = max(h1, h2) + 4
  top1 = (max_height - h1) // 2
  top2 = (max_height - h2) // 2
  color = (255, 255, 255)
  if axis == 1:
    padded_img1 = Image.new('RGB', (w1, max_height), color)
    padded_img2 = Image.new('RGB', (w2, max_height), color)

    padded_img1.paste(old_img, (0, top1))
    padded_img2.paste(new_img, (0, top2))
    im = np.array(padded_img1)
    im2 = np.array(padded_img2)
    im2 = np.concatenate((im2, im), axis=1)
    new_img = Image.fromarray(im2)
    return new_img
  else:
    padded_img1 = Image.new('RGB', (max_width, h1), color)
    padded_img2 = Image.new('RGB', (max_width, h2), color)
    padded_img1.paste(old_img, (0, 0))
    padded_img2.paste(new_img, (0, 0))
    im = np.array(padded_img1)
    im2 = np.array(padded_img2)
    im2 = np.concatenate((im2, im), axis=0)
    new_img = Image.fromarray(im2)
    return new_img

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)
