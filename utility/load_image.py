import os

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


def read_tif(image_path):

    ims = Image.open(image_path)

    img_array = []
    ret, images = cv2.imreadmulti(image_path, [], cv2.IMREAD_LOAD_GDAL)
    for i in range(len(images)):

        image = images[i]
        ims.seek(i)

        # resize with dpi
        resoultion = ims.info["dpi"]
        size = ims.size
        # 가로의 해상도가 더 클 경우
        if resoultion[1] < resoultion[0]:
            new_size = (int((size[0] / resoultion[0]) * resoultion[1]), size[1])
            image = cv2.resize(image, new_size)
        # 세로의 해상도가 더 클 경우
        elif resoultion[0] < resoultion[1]:
            new_size = (size[0], int((size[1] / resoultion[1]) * resoultion[0]))
            image = cv2.resize(image, new_size)
        else:
            pass

        # 만약 데이터가 (H,W) 형식이라면 3채널을 가지도록 보정
        if len(image.shape) <= 2:
            image = np.dstack([image, image, image])
        # 만약 데이터가 bool 형식이라면 0~255 값으로 보정
        if np.max(image) <= 1:
            image = image * 255
        img_array.append(image)

    return np.array(img_array)


def read_gif(image_path):
    im = cv2.VideoCapture(image_path)
    img_array = []

    while True:
        ret, frame = im.read()  # ret=True if it finds a frame else False.
        if ret == False:
            break
        img = Image.fromarray(frame)
        img = img.convert("RGB")
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_array.append(img)

    return np.array(img_array)


def read_pdf(image_path):
    img_array = []
    pdf_images = convert_from_path(image_path, dpi=150)
    for i in pdf_images:
        img = np.asarray(i)
        img_array.append(img)
    img_array = np.array(img_array)

    return img_array


def read_normal(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 만약 데이터가 (H,W) 형식이라면 3채널을 가지도록 보정
    if len(image.shape) <= 2:
        image = np.dstack([image, image, image])
    # 만약 데이터가 bool 형식이라면 0~255 값으로 보정
    if np.max(image) <= 1:
        image = image * 255

    image = np.array(image).astype(np.uint8)

    return np.array([image])


def load_image(image_path):
    """
    Input:

    image_path

    Output:

    np.uint8 type numpy array data

    """

    file_exts = os.path.basename(image_path).split(".")[-1].lower()

    """
    지원 가능한 확장자 목록
    tif / jpg / jpeg / png / bmp / gif / pdf
    """
    # 다중 tif 이미지 파일에 대응
    if (file_exts == "tif") or (file_exts == "tiff"):

        img_array = read_tif(image_path)

    elif file_exts == "gif":

        img_array = read_gif(image_path)

    elif file_exts == "pdf":

        img_array = read_pdf(image_path)

    else:

        img_array = read_normal(image_path)

    return img_array
