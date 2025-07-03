import cv2
import numpy as np
import math
import time
from paddleocr import PaddleOCR
from paddleocr import TextRecognition
from tool.public_info import device, text_ocr_type, det_true_list
from concurrent.futures import ThreadPoolExecutor
from tool.img_utils import convert_coordinates_restore

ocr = PaddleOCR(
    device=device,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir='models/PP-OCRv5_mobile_det_infer',
    text_recognition_model_dir='models/PP-OCRv5_mobile_rec_infer',
    enable_mkldnn=True,
    mkldnn_cache_capacity=5,  # 增加缓存容量
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
ocr_rec = TextRecognition(
    mkldnn_cache_capacity=2,
    enable_mkldnn=True,
    device=device, model_name='PP-OCRv5_mobile_rec',
    model_dir=r"models/PP-OCRv5_mobile_rec_infer")


def predict(img, lock='rec'):
    if lock == 'ocr':
        return ocr.predict(img)
    else:
        return ocr_rec.predict_iter(img)


def crop_and_preprocess_for_ocr(obj, orig_size, new_size, img, enable_det_list):
    """
    还原为原始图像的坐标并对图像指定区域裁剪

    参数:
        obj: 检测对象的坐标和标签信息
        orig_size: 原始图像尺寸 (width, height)
        new_size: 新的图像尺寸 (width, height)
        img: 原始图像数据
        enable_det_list: 需要进行检测增强的标签列表

    返回值:
        tuple: 裁剪后的图像、标签、是否需要检测增强
            - cropped_img: 处理后的裁剪图像
            - label: 对象的标签
            - is_det: 是否需要检测增强的布尔值
    """

    left, top, right, bottom = obj[0], obj[1], obj[2], obj[3]
    label = str(obj[4])

    # 坐标转换
    box = convert_coordinates_restore([left, top, right, bottom], orig_size, new_size)
    left = max(0, math.floor(box[0]))
    top = max(0, math.floor(box[1] - 1))
    right = min(img.shape[1], math.ceil(box[2] + 1))
    bottom = min(img.shape[0], math.ceil(box[3] + 1))

    # 检查有效区域
    if (right <= left) or (bottom <= top):
        return None, None, None

    # 裁剪图像
    cropped_img = img[top:bottom, left:right]
    if cropped_img.size <= 0:
        return None, None, None

    # 判断是否需要检测增强
    is_det = label in enable_det_list

    # 统一通道格式
    if len(cropped_img.shape) == 2:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    elif cropped_img.shape[2] == 4:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2RGB)
    else:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('v.png', cropped_img)
    if is_det:
        border_size = 2
        bordered_img = np.full((cropped_img.shape[0] + 2 * border_size, cropped_img.shape[1] + 2 * border_size, 3), 255,
                               dtype=np.uint8)
        bordered_img[border_size:-border_size, border_size:-border_size] = cropped_img
        # cropped_img = cv2.copyMakeBorder(
        #   cropped_img,
        #   top=5, bottom=5, left=5, right=5,
        #   borderType=cv2.BORDER_CONSTANT,
        #   value=[255, 255, 255]
        # )

    return cropped_img, label, is_det


executor = ThreadPoolExecutor(max_workers=2)


def process_and_group_detections(converted_detections, orig_size, new_size, img, enable_det_list):
    """
    处理并分组检测对象

    参数:
        converted_detections: 转换后的检测对象列表，包含坐标和标签信息
        orig_size: 原始图像尺寸 (width, height)
        new_size: 新的图像尺寸 (width, height)
        img: 原始图像数据
        enable_det_list: 需要进行检测增强的标签列表

    返回值:
        dict: 包含处理结果的字典，字段如下:
            - det_true_img: 需要检测增强的图像列表
            - det_true_label: 需要检测增强的标签列表
            - det_false_img: 不需要检测增强的图像列表
            - det_false_label: 不需要检测增强的标签列表
    """
    det_group = {"det_true_img": [], "det_true_label": [], "det_false_img": [], "det_false_label": []}
    for obj in converted_detections:
        cropped_img, label, is_det = crop_and_preprocess_for_ocr(obj, orig_size, new_size, img, enable_det_list)
        if cropped_img is not None:
            group_key = "det_true" if is_det else "det_false"
            det_group[group_key + "_img"].append(cropped_img)
            det_group[group_key + "_label"].append(label)
    return det_group


def ocr_and_set_value(converted_detections, img, new_size, ocr_result, orig_size):
    """
    对检测区域进行OCR识别并将结果设置到结果字典中

    参数:
        converted_detections: 转换后的检测对象列表，包含坐标和标签信息
        img: 原始图像数据
        new_size: 新的图像尺寸 (width, height)
        ocr_result: OCR结果存储字典
        orig_size: 原始图像尺寸 (width, height)

    返回值:
        无，直接修改ocr_result参数
    """
    start_time = time.perf_counter()
    result_groups = process_and_group_detections(converted_detections, orig_size, new_size, img, det_true_list)
    det_img = result_groups.get("det_false_img", [])
    det_label = result_groups.get("det_false_label", [])
    analysis_ocr_result(det_img, det_label, ocr_result)
    det_img = result_groups["det_true_img"]
    det_label = result_groups["det_true_label"]
    analysis_ocr_result(det_img, det_label, ocr_result, 'ocr', rec_text='rec_texts')

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"识别耗时: {execution_time} 秒")


def analysis_ocr_result(det_img, det_label, ocr_result, ocr='rec', rec_text='rec_text'):
    if det_img:
        if text_ocr_type == 1:
            batch_text_ocr(det_img, det_label, ocr, ocr_result, rec_text)
        else:
            text_ocr(det_img, det_label, ocr, ocr_result, rec_text)


def text_ocr(det_img, det_label, ocr, ocr_result, rec_text):
    futures = [
            executor.submit(
                text_predict,
                det_label, di, i, ocr, ocr_result, rec_text
            )
            for i, di in enumerate(det_img)
        ]
    for future in futures:
        future.result()

    # for i, di in enumerate(det_img):
    #     text_predict(det_label, di, i, ocr, ocr_result, rec_text)


def text_predict(det_label, di, i, ocr, ocr_result, rec_text):
    rr = predict(di, ocr)
    if rr:
        for index, line in enumerate(rr):
            if line:
                ocr_result.setdefault(det_label[i], '')
                ocr_result[det_label[i]] += ''.join(line[rec_text])


def batch_text_ocr(det_img, det_label, ocr, ocr_result, rec_text):
    # 批量识别
    rr = predict(det_img, ocr)
    if rr:
        for index, line in enumerate(rr):
            if line:
                ocr_result.setdefault(det_label[index], '')
                ocr_result[det_label[index]] += ''.join(line[rec_text])
