import json
import logging
import os

import cv2
import mmcv
import numpy as np
import yaml
from colorlog import ColoredFormatter
from shapely import geometry


def setup_logger(
    log_file_path: str = None,
    name: str = "TRBA_TPSearch",
    mode: str = "a",
    level=logging.INFO,
):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(filename)s"
        "[line:%(lineno)d]: %(message)s",
        # Define the format of the output log
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s"
        "[line:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )

    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if log_file_path is not None:
            if not os.path.exists(os.path.dirname(log_file_path)):
                os.makedirs(os.path.dirname(log_file_path))
            file_handler = logging.FileHandler(log_file_path, mode=mode)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False
    # logger.info('logger init finished')

    return logger


def draw_result(
    data_classes, img, polygons, classes, scores, out_file=None, color_list=None
):

    CLASS_dict = dict()
    for idx, c in enumerate(data_classes):
        CLASS_dict[c] = idx

    if color_list is None:
        color_mode = "random"
    else:
        color_mode = "fixed"

    # make img darker
    dark_page = np.full(img.shape, 50)
    img = img * 0.5 + dark_page * 0.5

    for polygon, clas, score in zip(polygons, classes, scores):

        if color_mode == "random":
            draw_color = np.random.randint(0, 225, 3, dtype=np.uint8)
        else:
            draw_color = np.array(color_list[CLASS_dict[clas]])

        label = f"{clas} {score}"

        img = draw_bbox(img, polygon, label, tuple(draw_color.tolist()))

    if out_file is None:
        return img
    else:
        mmcv.imwrite(img, out_file)


def get_bbox_and_polygon(points):
    """
    2point 또는 4point 박스의 좌표를 받아서 양쪽 모두의 좌표로 변환해주는 함수

    Input:
        points: [[x1,y1],[x2,y2]] or [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] shape의 array

    Output:
        box_point : [[x1,y1],[x2,y2]]
        polygon_point : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    """

    if len(points) == 4:

        points = np.array(points)

        x0 = np.min(points[:, 0])
        y0 = np.min(points[:, 1])
        x1 = np.max(points[:, 0])
        y1 = np.max(points[:, 1])

        bbox_point = [x0, y0, x1, y1]

        seg_points = points

    else:

        points = np.array(points)

        x0 = np.min(points[:, 0])
        y0 = np.min(points[:, 1])
        x1 = np.max(points[:, 0])
        y1 = np.max(points[:, 1])

        A = [x0, y0]
        B = [x0, y1]
        C = [x1, y1]
        D = [x1, y0]

        seg_points = [A, B, C, D]

        bbox_point = [x0, y0, x1, y1]

    return np.array(bbox_point), recoord(np.array(seg_points))


def recoord(polygon_coord):
    """
    4 point의 폴리곤을 받아 왼쪽 위의 좌표를 기준으로 시계방향으로 정렬해주는 함수

    Input:
        polygon_coord : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [4,2] shape의 array

    Output:
        polygon_coord : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [4,2] shape의 시계방향으로 정렬된 array

    """

    coord = np.array(polygon_coord)
    distance = []

    for idx, c in enumerate(coord):
        x = c[0]
        y = c[1]

        zero_distance = x + y
        distance.append([idx, zero_distance])

    distance = sorted(distance, key=lambda x: x[1])

    left_top = coord[distance[0][0]]
    right_bottom = coord[distance[-1][0]]

    box1 = coord[distance[1][0]]
    box2 = coord[distance[2][0]]

    if box1[0] > box2[0]:
        left_bottom = box2
        right_top = box1
    else:
        left_bottom = box1
        right_top = box2

    x1, y1 = left_top
    x2, y2 = left_bottom
    x3, y3 = right_bottom
    x4, y4 = right_top

    polygon = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    return np.array(polygon)


def read_json(config_path):
    """
    json 파일을 읽어 dict 형태로 반환해주는 함수
    """
    with open(config_path, encoding="utf-8-sig") as json_file:
        json_data = json.load(json_file)
    return json_data


def save_json(config, config_path):
    """
    dict 형태의 데이터를 받아 json파일로 저장해주는 함수
    """
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def read_yaml(config_path):
    """
    yaml 파일을 읽어 dict 형태로 반환해주는 함수
    """
    with open(config_path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    return data_dict


def save_yaml(config, config_path):
    """
    dict 형태의 파일을 받아 yaml 파일로 저장해주는 함수
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def change_polygon(original_shape, new_shape, single_polygon):
    """
    1번 이미지의 shape에 맞춰진 4 point polygon 박스의 좌표를 받아서
    2번 이미지의 shape에 같은 비율의 위치에 맞춰지게 변경해주는 함수

    Input:
        original_shape : 1번 이미지의 shape
        new_shape : 2번 이미지의 shape
        single_polygon : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [4,2] shape의 polygon box

    Output:

        new_coord: 2번 이미지에 맞게 변경된 4 point polygon
    """

    y_, x_ = original_shape

    target_y, target_x = new_shape
    x_scale = target_x / x_
    y_scale = target_y / y_

    # original frame as named values

    a, b, c, d = single_polygon

    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    new_x1 = x1 * x_scale
    new_y1 = y1 * y_scale

    new_x2 = x2 * x_scale
    new_y2 = y2 * y_scale

    new_x3 = x3 * x_scale
    new_y3 = y3 * y_scale

    new_x4 = x4 * x_scale
    new_y4 = y4 * y_scale

    new_coord = [
        [new_x1, new_y1],
        [new_x2, new_y2],
        [new_x3, new_y3],
        [new_x4, new_y4],
    ]

    return np.array(new_coord)


def change_box(original_shape, new_shape, single_box):

    """
    1번 이미지의 shape에 맞춰진 2 point box 박스의 좌표를 받아서
    2번 이미지의 shape에 같은 비율의 위치에 맞춰지게 변경해주는 함수

    Input:
        original_shape : 1번 이미지의 shape
        new_shape : 2번 이미지의 shape
        single_polygon : [[x1,y1],[x2,y2]], [2,2] shape의 box

    Output:

        new_coord: 2번 이미지에 맞게 변경된 2 point box
    """

    y_, x_ = original_shape

    target_y, target_x = new_shape
    x_scale = target_x / x_
    y_scale = target_y / y_

    # original frame as named values

    x0, y0, x1, y1 = single_box

    new_x0 = x0 * x_scale
    new_y0 = y0 * y_scale

    new_x1 = x1 * x_scale
    new_y1 = y1 * y_scale

    new_coord = [new_x0, new_y0, new_x1, new_y1]
    return np.array(new_coord)


def draw_bbox(
    img, point, label=None, color=(255, 0, 0), thickness=2, text_location="UP"
):
    """
    4 point polygon 을 받아 이미지에 그려주는 함수
    text를 넣을 경우 같이 표시해주는 기능

    Input:

        img: bbox가 그려질 이미지
        point : 4point polygon box
        label : polygon box 와 같이 표시될 text
        color : polygon box 의 RGB값
        thickness : polygon box 선 두께
        text_location :
            "UP" : polygon box의 왼쪽 위 점 위에 text표시
            "DOWN" : polygon box의 왼쪽 위 점 아래에 text표시
            "LEFT" : polygon box의 왼족 위 점 왼쪽에 text표시
            "RIGTH" : polygon box의 오른쪽 위 점 오른쪽에 text표시

    Output:

        img: bbox가 그려진 이미지
    """
    point = np.array(point).astype(int)
    cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)

    if label is not None:
        font_scale = img.shape[0] / 1000
        font = cv2.FONT_HERSHEY_SIMPLEX

        rectangle_bgr = (0, 0, 0)

        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(
            label, font, fontScale=font_scale, thickness=1
        )[0]

        if text_location == "UP":
            # set the text start position
            text_offset_x = point[0][0]
            text_offset_y = point[0][1] - text_height
        elif text_location == "DOWN":
            # set the text start position
            text_offset_x = point[0][0]
            text_offset_y = point[0][1] + text_height
        elif text_location == "LEFT":
            # set the text start position
            text_offset_x = point[0][0] - text_width
            text_offset_y = point[0][1]
        elif text_location == "RIGHT":
            # set the text start position
            text_offset_x = point[0][0] + (point[3][0] - point[0][0])
            text_offset_y = point[0][1]
        # make the coords of the box with a small padding of two pixels
        box_coords = (
            (text_offset_x, text_offset_y),
            (text_offset_x + text_width + 2, text_offset_y + text_height + 2),
        )
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(
            img,
            label,
            (text_offset_x, text_offset_y + text_height),
            font,
            fontScale=font_scale,
            color=color,
            thickness=1,
        )

    return img


def make_polygon(seg_map):
    """
    이진화된 mask map을 받아서 4point polygon을 생성해주는 함수

    Input:
        segmap : binary mask map

    Output:
        box : 4 point polygon box
    """

    seg_map = np.array(seg_map).astype(np.uint8) * 255

    # find contour
    # ret, thr = cv2.threshold(seg_map, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(seg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        big_cnt_area = 0
        big_cnt_area_idx = 0
        for idx, cnt in enumerate(contours):
            cnt_area = cv2.contourArea(cnt)

            if cnt_area > big_cnt_area:
                big_cnt_area = cnt_area
                big_cnt_area_idx = idx

        cnt = contours[big_cnt_area_idx]
    else:
        cnt = contours[0]

    # make 4 point polygon
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return np.array(box)


def get_files(root_dir: str, file_exts: list):
    image_list = []
    if os.path.isfile(root_dir):
        return [root_dir]

    for (path, dir, files) in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in file_exts:
                image_path = f"{path}/{filename}"
                image_list.append(image_path)

    return image_list


def fit_coord(target_coord, crop_coord, crop_size):
    # change coord to fit cropped_area
    target_coord = np.array(target_coord)

    if len(target_coord.shape) == 1:
        # coord type "box"
        fitted_coord = target_coord - np.array(
            [crop_coord[0], crop_coord[1], crop_coord[0], crop_coord[1]]
        )
    else:
        # coord type "polygon"
        fitted_coord = target_coord.copy()
        fitted_coord[:, 0] = fitted_coord[:, 0] - crop_coord[0]
        fitted_coord[:, 1] = fitted_coord[:, 1] - crop_coord[1]
    fitted_coord = np.clip(fitted_coord, a_min=0, a_max=crop_size[0])

    return fitted_coord


def create_json(input_image_path, original_image, polygons, labels):

    json_dict = dict()
    json_dict["version"] = "4.2.9"
    json_dict["flags"] = {}
    json_shapes = list()

    base_name = os.path.basename(input_image_path)
    height, width, _ = original_image.shape

    # polygons = predictions["polygons"]
    # labels = predictions["labels"]

    for polygon, label_name in zip(polygons, labels):

        new_label = dict()
        new_label["label"] = label_name
        new_label["points"] = polygon.tolist()
        new_label["group_id"] = None
        new_label["shape_type"] = "polygon"
        new_label["flags"] = {}
        json_shapes.append(new_label)

    json_dict["shapes"] = json_shapes
    json_dict["imagePath"] = base_name
    json_dict["imageData"] = None
    json_dict["imageHeight"] = height
    json_dict["imageWidth"] = width

    return json_dict


def polygon_IOU(polygon_1, polygon_2):
    """
    Calculate the IOU of two polygons
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: Same as above
    :return:
    """
    polygon1 = geometry.Polygon(polygon_1).buffer(0)
    polygon2 = geometry.Polygon(polygon_2).buffer(0)

    intersection = polygon1.intersection(polygon2).area

    union = polygon1.union(polygon2).area

    if union == 0:
        return 0
    else:
        return intersection / union


def get_center(polygon):

    A, B, C, D = polygon

    x0, y0 = A
    x1, y1 = C

    w = x1 - x0
    h = y1 - y0

    center_coord = [x0 + w // 2, y0 + h // 2]

    return center_coord


def check_center(polygon, center_point):

    A, B, C, D = polygon

    x0, y0 = A
    x1, y1 = C

    cx, cy = center_point

    if (x0 < cx) and (x1 > cx) and (y0 < cy) and (y1 > cy):

        center = True
    else:
        center = False

    return center


def refine_result(predictions):

    pd_boxes = predictions["boxes"]
    pd_polygons = predictions["polygons"]
    pd_scores = predictions["scores"]
    predictions["labels"]

    refined_predictions = predictions.copy()
    delete_idx = []

    for fidx in range(len(pd_boxes)):
        f_polygon = pd_polygons[fidx]
        for bidx in range(fidx + 1, len(pd_boxes)):
            b_polygon = pd_polygons[bidx]
            iou_value = polygon_IOU(f_polygon, b_polygon)

            if iou_value <= 0:
                continue
            elif iou_value >= 0.3:
                f_score = pd_scores[fidx]
                b_score = pd_scores[bidx]

                if f_score > b_score:
                    delete_idx.append(bidx)
                elif f_score < b_score:
                    delete_idx.append(fidx)
                else:
                    pass
            else:
                pass

    delete_idx = sorted(list(np.unique(delete_idx)))
    # print(delete_idx)

    for didx in reversed(delete_idx):

        refined_predictions["boxes"].pop(didx)
        refined_predictions["polygons"].pop(didx)
        refined_predictions["scores"].pop(didx)
        refined_predictions["labels"].pop(didx)
        refined_predictions["classes"].pop(didx)

    return refined_predictions


def resize_and_pad_with_polygon(img, polygons, size, padcolor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # first stage polygon resize
    new_polys = []
    for poly in polygons:
        changed_poly = change_polygon((h, w), size, poly)

        new_polys.append(changed_poly)

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h
    new_polys_2 = []

    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

        for new_poly in new_polys:
            h_aspect = new_h / sh
            h_mvment = (sh - new_h) / 2

            new_poly = np.array(new_poly) * np.array([1, h_aspect])
            new_poly = new_poly + np.array([0, h_mvment])

            new_polys_2.append(new_poly)

    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(
            int
        )
        pad_top, pad_bot = 0, 0

        for new_poly in new_polys:
            w_aspect = new_w / sw
            w_mvment = (sw - new_w) / 2

            new_poly = np.array(new_poly) * np.array([w_aspect, 1])
            new_poly = new_poly + np.array([w_mvment, 0])

            new_polys_2.append(new_poly)

    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
        new_polys_2 = new_polys

    # set pad color
    if len(img.shape) == 3 and not isinstance(
        padcolor, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
        padColor = [padcolor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padColor,
    )

    return scaled_img, np.array(new_polys_2)


def restore_box_ratio(img, boxes, size):

    new_boxes = []

    h, w = img.shape[:2]
    sh, sw = size

    aspect = w / h

    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

        for box in boxes:

            new_box = np.array(box) + np.array([0, -pad_top, 0, -pad_top])

            new_box = change_box((sh - pad_top * 2, sw), (h, w), new_box)

            new_boxes.append(new_box)

    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(
            int
        )
        pad_top, pad_bot = 0, 0

        for box in boxes:

            new_box = np.array(box) + np.array([-pad_left, 0, -pad_left, 0])

            new_box = change_box((sh, sw - pad_left * 2), (h, w), new_box)

            new_boxes.append(new_box)

    else:
        for box in boxes:
            new_box = change_box((sh, sw), (h, w), box)
            new_boxes.append(new_box)

    return new_boxes


def restore_mask_ratio(img, masks, size):

    new_masks = []

    h, w = img.shape[:2]
    sh, sw = size

    aspect = w / h

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

        for mask in masks:

            new_mask = mask[pad_top : (sh - pad_bot), :]

            new_mask = cv2.resize(new_mask, (w, h), interpolation=interp)

            new_masks.append(new_mask)

    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(
            int
        )
        pad_top, pad_bot = 0, 0

        for mask in masks:

            new_mask = mask[:, pad_left : (sw - pad_right)]

            new_mask = cv2.resize(new_mask, (w, h), interpolation=interp)

            new_masks.append(new_mask)

    else:
        for mask in masks:

            new_mask = cv2.resize(mask, (w, h), interpolation=interp)
            new_masks.append(new_mask)

    return new_masks
