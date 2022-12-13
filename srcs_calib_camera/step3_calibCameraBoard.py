import os
import cv2
import numpy as np
from Calib import detect_circles, calib_camera
from utils import load_config_board


def calib(data_folder, calib_num, config_file, out_folder):
    calib_folder = os.path.join(out_folder, data_folder.split("/")[-1], "calib")

    config = load_config_board(config_file)
    rows = config["rows"]
    cols = config["cols"]
    square_size = config["square_size"]
    pattern_size = (cols, rows)

    pts3d_world = np.zeros((cols * rows, 3), np.float32)
    pts3d_world[:, :2] = np.mgrid[0: cols, 0: rows].T.reshape(-1, 2)
    pts3d_world *= square_size

    pts3d_world_all = []
    pts2d_circle_all = []

    image_size = None
    for i in range(calib_num):
        file = os.path.join(data_folder, str(i + 1) + ".bmp")
        img = cv2.imread(file, 0)
        if img is None:
            print("Failed to load the image:", file)
        if image_size is None:
            image_size = img.shape
        found_circle, pts2d_circle = detect_circles(img, pattern_size, True)
        if found_circle:
            pts3d_world_all.append(pts3d_world)
            pts2d_circle_all.append(pts2d_circle)

    calib_result_circle = calib_camera("circle-board",
        pts3d_world_all, pts2d_circle_all, image_size, config["flag_calib"], calib_folder)


if __name__ == '__main__':
    calib_num   = 20                            # 次数
    config_file = "./data/config_board.xml"     # 标定
    data_folder = "E:/calib/data/d1-actual"     # 数据文件夹
    out_folder  = "./out/calib"                 # 输出文件夹
    calib(data_folder, calib_num, config_file, out_folder)
