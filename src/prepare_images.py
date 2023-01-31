from typing import Final

import cv2
import numpy as np

from fruits_recognition import get_bounding


rows: Final[int] = 10
cols: Final[int] = 17

if __name__ == "__main__":
    img = cv2.imread("../images/sample.png")
    x, y, w, h = get_bounding(img)
    cropped_img = img[x : x + w, y : y + h]

    chunks = [[] for _ in range(rows)]
    for i, row_img in enumerate(np.array_split(cropped_img, rows, axis=0)):
        for j, chunk in enumerate(np.array_split(row_img, cols, axis=1)):
            chunks[i].append(chunk)

    x_index, y_index = 5, 2
    num = 6
    x, y, w, h = get_bounding(chunks[y_index][x_index])
    print(x, y, w, h)
    cropped_img = chunks[y_index][x_index][x : x + w, y : y + h]
    cv2.imwrite(f"{num}.jpg", cropped_img)
