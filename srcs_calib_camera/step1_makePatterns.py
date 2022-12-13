import os

import cv2
import numpy as np
from Phaser import Phaser
from GrayCode import GrayCode

from utils import load_config_screen, imwrite2


def make_chess_board(width, height, cols, rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cols = cols + 3
    rows = rows + 3
    chessboard = np.zeros((height, width), dtype=np.uint8)
    square_size_x = width / cols
    square_size_y = height / rows
    if square_size_x == square_size_y:
        square_size = int(square_size_x)
        org_X = int((height - rows * square_size) / 2)  # pattern关于纵轴方向的位置，默认放在中间
        org_Y = int((width - cols * square_size) / 2)   # pattern关于横轴方向的位置，默认放在中间
        color1 = 1
        img = np.zeros((int(rows * square_size), int(cols * square_size)), dtype=np.bool)
        for row in range(rows):
            color2 = color1
            for col in range(cols):
                if color2 == 1:
                    img[row * square_size: (row + 1) * square_size, col * square_size: (col + 1) * square_size] = color2
                color2 = 1 - color2
            color1 = 1 - color1
        chessboard[org_X: org_X + rows * square_size, org_Y: org_Y + cols * square_size] = img
        chessboard = (1 - chessboard) * 255
        # 四周空白
        chessboard[:square_size, :] = 255
        chessboard[height - square_size:, :] = 255
        chessboard[:, :square_size] = 255
        chessboard[:, width - square_size:] = 255

        cv2.imshow("chessboard", chessboard)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        imwrite2(save_dir, chessboard)


# 圆环标定板（圆心和角点相差0.5个像素）
def make_circle_board(width, height, cols, rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cols += 3
    rows += 3
    img = np.zeros((height, width), dtype=np.uint8)
    square_size_x = width / cols
    square_size_y = height / rows

    if square_size_x == square_size_y:
        square_size = square_size_x
        radius = int(square_size / 2 * 7 / 8)   # 半径
        print(radius)
        # 如果半径也是整数
        xs = np.arange(2 * square_size, width - 1 * square_size, square_size, dtype=np.int)
        ys = np.arange(2 * square_size, height- 1 * square_size, square_size, dtype=np.int)
        for y in ys:
            for x in xs:
                cv2.circle(img, (x, y), radius, (255, 255, 255), -1, 8)
        # 白边
        img[0: int(square_size * 1.5), :] = 255
        img[int(height - square_size * 1.5): height, :] = 255
        img[:, 0: int(square_size * 1.5)] = 255
        img[:, int(width - square_size * 1.5): width] = 255

        cv2.imshow("img", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        imwrite2(save_dir, img)


# 相移靶标
def make_phase_board(d, N, W, H, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n = d["n"]
    A = d["A"]
    B = d["B"]
    phaser = Phaser()
    graycode = GrayCode()

    # X方向条纹
    T_W = W / (2 ** n)
    phaser.makePatterns(A, B, N, T_W, W, H, save_dir, False)
    graycode.makePatterns(n, W, H, save_dir, False)

    # Y方向条纹
    T_H = H / (2 ** n)
    phaser.makePatterns(A, B, N, T_H, H, W, save_dir, True)
    graycode.makePatterns(n, H, W, save_dir, True)


if __name__ == '__main__':
    config_file = "data/config_screen.xml"
    save_dir = "out/patterns/"
    d = load_config_screen(config_file)
    W = d["Screen_Width_Pixel"]
    H = d["Screen_Height_Pixel"]
    cols = d["cols"]
    rows = d["rows"]
    N_max = 32   # 生成全部条纹图案

    # 01 绘制棋盘格
    make_chess_board(W, H, cols, rows, save_dir)
    # 02 绘制圆
    make_circle_board(W, H, cols, rows, save_dir)
    # 03 相移法
    make_phase_board(d, N_max, W, H, save_dir)
    # 用于二次设置背景
    make_chess_board(W, H, cols, rows, save_dir)






