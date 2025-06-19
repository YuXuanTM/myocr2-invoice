import os
from paddleocr import PaddleOCR
from paddleocr import TextRecognition
from tool.public_info import device

ocr = PaddleOCR(
    device = device,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir='models/PP-OCRv5_mobile_det_infer',
    text_recognition_model_dir='models/PP-OCRv5_mobile_rec_infer',
    # enable_mkldnn=True,
    # mkldnn_cache_capacity=2048,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
ocr_rec = TextRecognition(device = device,model_name = 'PP-OCRv5_mobile_rec', model_dir=r"models/PP-OCRv5_mobile_rec_infer")


def predict(img, lock ='rec'):
  if lock == 'ocr':
    return ocr.predict(img, )
  else:
    return ocr_rec.predict_iter(img)