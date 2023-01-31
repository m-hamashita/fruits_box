# for cv2 stab: https://github.com/opencv/opencv/issues/14590#issuecomment-1133515901
import numpy as np
import cv2
from typing import Final

rows: Final[int] = 10
cols: Final[int] = 17


def get_bounding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([18, 0, 0])
    upper_color = np.array([48, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 検出した色が含まれる領域を抽出する
    coords = np.column_stack(np.where(mask > 0))
    rect = cv2.boundingRect(coords)

    return rect


def get_number(img, templates):
    scores = np.zeros(9)
    for i in range(9):
        result = cv2.matchTemplate(img, templates[i], cv2.TM_CCOEFF_NORMED)
        scores[i] = result.max()
    return np.argmax(scores) + 1


def print_fruits(number_list):
    for number in number_list:
        print(number)


def get_fruits_num(img):
    x, y, w, h = get_bounding(img)
    cropped_img = img[x : x + w, y : y + h]
    cropped_img = cv2.resize(cropped_img, (1108, 650))

    templates = []
    for i in range(1, 10):
        templates.append(cv2.imread(f"../images/{i}.jpg"))

    chunks = [[] for _ in range(rows)]
    for i, row_img in enumerate(np.array_split(cropped_img, rows, axis=0)):
        for _, chunk in enumerate(np.array_split(row_img, cols, axis=1)):
            chunks[i].append(chunk)

    fruits_num = [[] for _ in range(rows)]
    for i, row in enumerate(chunks):
        for _, chunk in enumerate(row):
            fruits_num[i].append(get_number(chunk, templates))

    return fruits_num


if __name__ == "__main__":
    img = cv2.imread("../images/resize_small.png")
    fruits_num = get_fruits_num(img)

    print_fruits(fruits_num)
