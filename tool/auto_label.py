import cv2
import os
import json
import numpy as np
import predict2


from img_utils import img_joint, resize_size_maintain_aspect_ratio, convert_coordinates_restore, __get_img__, generate_pretty_color
from PIL import Image


# 自动标注
# 使用PPOCRLabel打开target_path路径进行微调
def label(public_info):
    files = os.listdir(public_info.directory)

    label_file_name = public_info.target_path + "/Label.txt"
    existing_lines = []
    try:
        file = open(label_file_name, 'r+', encoding='utf-8')
        for readline in file.readlines():
            existing_lines.append(readline.split('\t')[0])
    except FileNotFoundError:
        file = open(label_file_name, 'w', encoding='utf-8')

    for file_name in files:
        # count = count + 1
        name = file_name.split('.')[0]
        file_path = os.path.join(public_info.directory, file_name)
        imgs = __get_img__(file_name, file_path)
        index = 0
        for img in imgs:
            index += 1
            processed_img, orig_size, new_size, paste_coords, canvas = resize_size_maintain_aspect_ratio(img, 640, 640)
            name = name + str(index)
            img_file = public_info.target_path + "/1_" + f"{name}" + ".png"
            label_img_url = public_info.label_img_url + "/1_" + f"{name}" + ".png"
            # 保存图片, 缩放后的.
            # canvas.save(img_file)
            Image.fromarray(img).save(img_file)
            # cv2.imwrite(imgfile, canvas)
            converted_detections, item_infos, item_boxes, items = predict2.start(canvas)
            converted_detections = list(converted_detections)
            for item in items:
                converted_detections.append(list(item))
            for item in item_boxes:
                converted_detections.append(item)
            canvas = np.array(canvas)
            # results = model(processed_img)[0]
            # boxes = results.boxes.data.tolist()
            # names = results.names
            data_line = []
            labels = {}
            for obj in converted_detections:
                left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                label = str(obj[4])
                confidence = obj[5]
                if label in labels:
                    continue
                color = generate_pretty_color()
                # labels[label] = label
                cv2.rectangle(canvas, (left, top), (right, bottom), color=color, thickness=1, lineType=cv2.LINE_AA)
                caption = f"{label} {confidence:.2f}"
                w, h = cv2.getTextSize(caption, 0, 0.5, 1)[0]
                box = convert_coordinates_restore([left, top, right, bottom], orig_size, new_size, paste_coords)
                o_left, o_top, o_right, o_bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                img = np.array(img)
                cv2.rectangle(img, (o_left - 1, o_top - 20), (o_left + w + 10, o_top), color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(img, caption, (o_left, o_top - 5), 0, 0.5, (0, 0, 0), 1, 2)
                cv2.rectangle(img, (o_left, o_top), (o_right, o_bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
                # points = [[left, top], [right, top], [right, bottom], [left, bottom]]
                points = [[o_left, o_top], [o_right, o_top], [o_right, o_bottom], [o_left, o_bottom]]
                data = {
                    "transcription": label,
                    "points": points,
                    "difficult": False
                }
                data_line.append(data)
            json_data = json.dumps(data_line)
            # 保存对比图.
            img_joint(Image.fromarray(img), Image.fromarray(canvas), 1).save(public_info.contrast_path + "/" + f"{name}" + ".png")
            aa = label_img_url + "	" + json_data
            print(img_file)
            if label_img_url not in existing_lines:
                # 写入标注位置信息.
                file.write(aa + '\n')
