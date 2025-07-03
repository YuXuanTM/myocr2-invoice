import numpy as np
import predict2

from tool.public_info import types
from tool.img_utils import resize_to_fixed_size, __get_img__
from TextOCR import ocr_and_set_value
from flask import Flask, request
from PIL import Image, ImageEnhance

app = Flask(__name__)
app.json.ensure_ascii = False


@app.route('/invoice_ocr', methods=['POST'])
def invoice_ocr():
    uploaded_file = request.files['file']
    print(uploaded_file.content_type)
    if uploaded_file is None or uploaded_file.content_type not in types:
        return {}
    read = uploaded_file.read()
    filename = uploaded_file.filename
    ocr_result = {}
    imgs = __get_img__(filename, read)
    for img in imgs:
        img2 = Image.fromarray(img)
        processed_img, orig_size, new_size, paste_coords, resize_img = resize_to_fixed_size(img2)
        converted_detections, item_infos, item_boxes, items = predict2.start(resize_img)
        enhancer = ImageEnhance.Contrast(img2)
        img2 = enhancer.enhance(2.0).convert('L')
        img = np.array(img2)
        ocr_and_set_value(converted_detections, img, new_size, ocr_result, orig_size)
        item_result_list = []
        for item in item_infos:
            item_result = {}
            item_result_list.append(item_result)
            ocr_and_set_value(item, img, new_size, item_result, orig_size)
        if item_result_list.__len__() > 0:
            ocr_result['items'] = item_result_list

    return ocr_result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
