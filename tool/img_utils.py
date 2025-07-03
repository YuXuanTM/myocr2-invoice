import cv2
import base64
import os
import numpy as np
import fitz
import random

from PIL import Image
from torchvision import transforms
from easyofd.ofd import OFD

FIXED_WIDTH = 2500


def __get_img__(filename, file):
    """
    从不同格式的文件中提取图像数据。

    参数:
        filename (str): 文件名。
        file (Union[str, bytes]): 文件内容，可以是字符串或字节流。

    返回:
        list: 包含图像数据的列表。
    """
    filename = filename.lower()
    imgs = []
    if filename.endswith('.ofd'):
        ofd = OFD()
        if isinstance(file, str):
            with open(file, "rb") as f:
                ofdb64 = str(base64.b64encode(f.read()), "utf-8")
        else:
            ofdb64 = str(base64.b64encode(file), "utf-8")
        ofd.read(ofdb64, save_xml=False,
                 xml_name=f"{os.path.split(filename)[0]}_xml")
        img_np = ofd.to_jpg()
        ofd.del_data()
        for index in range(img_np.__len__()):
            img = np.array(img_np[index])
            imgs.append(img)
    elif filename.endswith('.pdf'):
        if isinstance(file, str):
            doc = fitz.open(file)
        else:
            doc = fitz.open("pdf", file)
        if len(doc) == 0:
            return {}
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            original_width = page.rect.width
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                (w, h) = img[2], img[3]
                if w / original_width > 0.5:
                    continue
                doc._deleteObject(xref)
            scale_factor = FIXED_WIDTH / original_width
            pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        # processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
    else:
        if isinstance(file, str):
            # img = cv2.imread(file)[:, :, ::-1]
            img = np.array(Image.open(file))
        else:
            img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.COLOR_BGR2RGB)
        imgs.append(img)
        # processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
    return imgs


def convert_coordinates_restore(box, orig_size, new_size, paste_coords=(0, 0)):
    """
    将坐标从新尺寸映射回原始尺寸。

    参数:
        box (list): 包含四个坐标的列表 [x1, y1, x2, y2]。
        orig_size (tuple): 原始图像的尺寸 (width, height)。
        new_size (tuple): 新图像的尺寸 (width, height)。
        paste_coords (tuple): 图像粘贴到画布上的偏移量 (paste_x, paste_y)，默认为 (0, 0)。

    返回:
        list: 映射回原始尺寸的坐标 [x1_new, y1_new, x2_new, y2_new]。
    """
    orig_w, orig_h = orig_size
    new_w, new_h = new_size
    paste_x, paste_y = paste_coords

    x1, y1, x2, y2 = box
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    x1_new = (x1 - paste_x) * scale_x
    y1_new = (y1 - paste_y) * scale_y
    x2_new = (x2 - paste_x) * scale_x
    y2_new = (y2 - paste_y) * scale_y

    return [x1_new, y1_new, x2_new, y2_new]


def convert_coordinates_zoom(box, orig_size, new_size, paste_coords):
    """
    将坐标从原始尺寸缩放到新尺寸。

    参数:
        box (list): 包含四个坐标的列表 [x1, y1, x2, y2]。
        orig_size (tuple): 原始图像的尺寸 (width, height)。
        new_size (tuple): 新图像的尺寸 (width, height)。
        paste_coords (tuple): 图像粘贴到画布上的偏移量 (paste_x, paste_y)。

    返回:
        list: 缩放到新尺寸的坐标 [x1_new, y1_new, x2_new, y2_new]。
    """
    orig_w, orig_h = orig_size
    new_w, new_h = new_size
    paste_x, paste_y = paste_coords

    x1, y1, x2, y2 = box
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    x1_new = x1 * scale_x + paste_x
    y1_new = y1 * scale_y + paste_y
    x2_new = x2 * scale_x + paste_x
    y2_new = y2 * scale_y + paste_y

    return [x1_new, y1_new, x2_new, y2_new]


def resize_size_maintain_aspect_ratio(image_path, target_w=640, target_h=640):
    """
    保持纵横比调整图片大小并转换为张量格式。

    参数:
        image_path (np.ndarray): 输入的图像数据，以NumPy数组形式表示。
        target_w (int): 目标宽度，默认为640。
        target_h (int): 目标高度，默认为640。

    返回:
        tuple: 包含以下内容的元组：
            - input_tensor (torch.Tensor): 转换后的图像张量，形状为 [1, C, H, W]。
            - orig_size (tuple): 原始图像的尺寸 (width, height)。
            - new_size (tuple): 新图像的尺寸 (width, height)。
            - paste_coords (tuple): 图像粘贴到画布上的偏移量 (paste_x, paste_y)。
            - canvas (PIL.Image): 处理后的图像对象。
    """
    img = Image.fromarray(image_path)
    orig_w, orig_h = img.size
    if orig_w > orig_h:
        new_w = target_w
        new_h = int(orig_h * (new_w / orig_w))
    else:
        new_h = target_h
        new_w = int(orig_w * (new_h / orig_h))

    # Resize the image
    img_resized = img.resize((new_w, new_h))

    # Create a black canvas with the target size
    canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
    # Paste the resized image onto the canvas
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    transform = transforms.ToTensor()
    input_tensor = transform(canvas).unsqueeze(0)
    return input_tensor, (orig_w, orig_h), (new_w, new_h), (paste_x, paste_y), canvas


def resize_to_fixed_size(img, target_w=640, target_h=640):
    """
    对图像进行预处理，调整大小并转换为张量格式。

    参数:
        img (Union[np.ndarray, PIL.Image]): 输入的图像数据，可以是NumPy数组或PIL图像对象。
        target_w (int): 目标宽度，默认为512。
        target_h (int): 目标高度，默认为512。

    返回:
        tuple: 包含以下内容的元组：
            - input_tensor (torch.Tensor): 转换后的图像张量，形状为 [1, C, H, W]。
            - orig_size (tuple): 原始图像的尺寸 (width, height)。
            - new_size (tuple): 新图像的尺寸 (width, height)。
            - paste_coords (tuple): 图像粘贴到画布上的偏移量 (paste_x, paste_y)，默认为 (0, 0)。
            - resize_img (PIL.Image): 调整大小后的图像对象。
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    orig_w, orig_h = img.size
    img_resized = img.resize((target_w, target_h))
    transform = transforms.ToTensor()
    input_tensor = transform(img_resized).unsqueeze(0)
    return input_tensor, (orig_w, orig_h), img_resized.size, (0, 0), img_resized


def img_joint(new_img, old_img, axis=0):
    w1, h1 = old_img.size
    w2, h2 = new_img.size
    max_width = max(w1, w2) + 4
    max_height = max(h1, h2) + 4
    top1 = (max_height - h1) // 2
    top2 = (max_height - h2) // 2
    color = (0, 0, 0)
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


def generate_pretty_color():
    # 避免过亮或过暗的颜色，这里我们设置RGB每个分量的范围为[64, 191]
    min_val = 64
    max_val = 191

    # 确保颜色有一定的饱和度，通过保持RGB值之间的差异
    r = random.randint(min_val, max_val)
    g = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)

    # 增加一些变化，以确保颜色不是单调的
    if max(r, g, b) - min(r, g, b) < 64:  # 如果RGB值之间的差异小于64，则调整其中一个值
        which_to_adjust = random.choice(['r', 'g', 'b'])
        adjustment = random.choice([-64, 64])  # 随机决定是增加还是减少
        if which_to_adjust == 'r':
            r = max(min(r + adjustment, max_val), min_val)
        elif which_to_adjust == 'g':
            g = max(min(g + adjustment, max_val), min_val)
        else:
            b = max(min(b + adjustment, max_val), min_val)

    return r, g, b
