import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 以灰度格式读取图像
def imread2(filename):
    img = cv2.imread(filename, flags=0)
    if img is None:
        raise FileNotFoundError("文件没有找到" + filename)
    else:
        return img


def expand_dim(ns):
    ns = np.expand_dims(ns, axis=1)
    ns = np.expand_dims(ns, axis=2)
    return ns


# 检测轮廓
def detectContour(img):
    # 01 阈值化
    ret, img_bin = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    # 02 找出轮廓
    mask_temp, contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 03 找出最大轮廓
    areas = [cv2.contourArea(contour) for contour in contours]
    max_idx = np.argmax(np.array(areas))
    # 04 生成掩模（最大圈处为1）
    mask = cv2.drawContours(np.zeros_like(img), contours, max_idx, 1, cv2.FILLED)
    # 05 腐蚀：减小白色区域
    mask = cv2.erode(mask, np.ones((5, 5)), iterations=5)
    # 06 展示图像
    img_show = np.stack((img,) * 3, axis=-1)
    img_show = cv2.drawContours(img_show, contours, max_idx, (0, 0, 255), thickness=3)
    img_show = cv2.resize(img_show, dsize=None, fx=0.4, fy=0.4)
    cv2.imshow("contours", img_show)
    cv2.waitKey(0)
    cv2.destroyWindow("contours")
    return mask


# 保存图像，并全局计数+1
global_idx = 0


def imwrite2(save_dir, img):
    os.makedirs(save_dir, exist_ok=True)
    # 用全局变量
    global global_idx
    global_idx += 1
    filename = os.path.join(save_dir, str(global_idx) + ".bmp")
    cv2.imwrite(filename, img)
    print("保存图像:", filename)


def load_config_board(filename):
    file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    node = file.getFirstTopLevelNode()
    BoardSize_Width = int(node.getNode("BoardSize_Width").real())
    BoardSize_Height = int(node.getNode("BoardSize_Height").real())
    BoardSize_SquareSize = float(node.getNode("BoardSize_SquareSize").real())

    aspectRatio = bool(node.getNode("Calibrate_FixAspectRatio").real())
    # 切向失真系数设置为零，并保持为零
    calibZeroTangentDist = bool(node.getNode("Calibrate_AssumeZeroTangentialDistortion").real())
    # 在全局优化期间不改变主点
    calibFixPrincipalPoint = bool(node.getNode("Calibrate_FixPrincipalPointAtTheCenter").real())
    # 径向畸变的 k1 设置为零
    calibZerok1Dist = bool(node.getNode("Calibrate_AssumeZerok1Distortion").real())
    # 径向畸变的 k2 设置为零
    calibZerok2Dist = bool(node.getNode("Calibrate_AssumeZerok2Distortion").real())
    # 径向畸变的 K3 设置为零
    calibZerok3Dist = bool(node.getNode("Calibrate_AssumeZerok3Distortion").real())

    flag_calib = 0
    if aspectRatio:
        flag_calib |= cv2.CALIB_FIX_ASPECT_RATIO
    if calibZeroTangentDist:
        flag_calib |= cv2.CALIB_ZERO_TANGENT_DIST
    if calibFixPrincipalPoint:
        flag_calib |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if calibZerok1Dist:
        flag_calib |= cv2.CALIB_FIX_K1
    if calibZerok2Dist:
        flag_calib |= cv2.CALIB_FIX_K2
    if calibZerok3Dist:
        flag_calib |= cv2.CALIB_FIX_K3
    d = {
        "rows": BoardSize_Height,
        "cols": BoardSize_Width,
        "flag_calib": flag_calib,
        "square_size": BoardSize_SquareSize,
    }
    return d


def load_config_screen(filename):
    file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    node = file.getFirstTopLevelNode()
    # 相移参数
    N = int(node.getNode("N").real())     # 相移步数
    B_min = node.getNode("B_min").real()  # 最小阈值
    n = int(node.getNode("n").real())
    A = node.getNode("A").real()
    B = node.getNode("B").real()
    iter_num = int(node.getNode("iter_num").real())
    # 屏幕参数
    inch  = node.getNode("inch").real()
    Screen_Size = node.getNode("Screen_Size").real()
    Screen_Width_Ratio = node.getNode("Screen_Width_Ratio").real()
    Screen_Height_Ratio = node.getNode("Screen_Height_Ratio").real()
    Screen_Width_Pixel = int(node.getNode("Screen_Width_Pixel").real())
    Screen_Height_Pixel = int(node.getNode("Screen_Height_Pixel").real())
    BoardSize_Width = int(node.getNode("BoardSize_Width").real())
    BoardSize_Height = int(node.getNode("BoardSize_Height").real())
    # 单个棋盘格的像素大小
    square_size_pixel = int(Screen_Width_Pixel / (BoardSize_Width + 3))
    # 计算单个p的大小
    Screen_Width = Screen_Size * inch * Screen_Width_Ratio / np.sqrt(Screen_Height_Ratio ** 2 + Screen_Width_Ratio ** 2)
    p = Screen_Width / Screen_Width_Pixel
    square_size = p * square_size_pixel

    # 标定flag
    # 固定长宽比（仅将 fy 作为自由参数，fx/fy 的比率与输入 cameraMatrix 中的比率相同）
    aspectRatio = bool(node.getNode("Calibrate_FixAspectRatio").real())
    # 切向失真系数设置为零，并保持为零
    calibZeroTangentDist = bool(node.getNode("Calibrate_AssumeZeroTangentialDistortion").real())
    # 在全局优化期间不改变主点
    calibFixPrincipalPoint = bool(node.getNode("Calibrate_FixPrincipalPointAtTheCenter").real())
    # 径向畸变的 k1 设置为零
    calibZerok1Dist = bool(node.getNode("Calibrate_AssumeZerok1Distortion").real())
    # 径向畸变的 k2 设置为零
    calibZerok2Dist = bool(node.getNode("Calibrate_AssumeZerok2Distortion").real())
    # 径向畸变的 K3 设置为零
    calibZerok3Dist = bool(node.getNode("Calibrate_AssumeZerok3Distortion").real())

    flag_calib = 0
    if aspectRatio:
        flag_calib |= cv2.CALIB_FIX_ASPECT_RATIO
    if calibZeroTangentDist:
        flag_calib |= cv2.CALIB_ZERO_TANGENT_DIST
    if calibFixPrincipalPoint:
        flag_calib |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if calibZerok1Dist:
        flag_calib |= cv2.CALIB_FIX_K1
    if calibZerok2Dist:
        flag_calib |= cv2.CALIB_FIX_K2
    if calibZerok3Dist:
        flag_calib |= cv2.CALIB_FIX_K3
    d = {
        "N": N,
        "B_min": B_min,
        "n": n,
        "A": A,
        "B": B,
        "iter_num": iter_num,
        "rows": BoardSize_Height,
        "cols": BoardSize_Width,
        "square_size": square_size,
        "flag_calib": flag_calib,
        "Screen_Width_Pixel": Screen_Width_Pixel,
        "Screen_Height_Pixel": Screen_Height_Pixel,
        "p": p
    }

    return d


    # plt.show()